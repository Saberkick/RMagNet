"""Joint student LoRA/SMA with an independent, immutable M4 teacher adapter."""
import copy
from contextlib import contextmanager
import torch
import safetensors.torch
from safetensors import safe_open
from .m4_cache import atomic_safetensors
from .sma import SMA_VERSION

JOINT_VERSION='sma-joint-b-v1'
TEACHER='sma_fixed_m4_teacher'

def canonical(name):return name.replace('.original.','.')

def lora_parameters(backend):
    return [p for n,p in backend.transformer.named_parameters() if '.lora_' in n and '.default.' in n]

def install_teacher(backend):
    state={canonical(n):p.detach().cpu().clone() for n,p in backend.transformer.named_parameters() if '.lora_' in n and '.default.' in n}
    backend.fixed_m4_teacher_state=state
    backend.set_trainable_branch('transmission')
    return {'storage':'CPU immutable M4 LoRA snapshot, staged weight switching',
        'tensors':len(state),'parameters':sum(p.numel() for p in state.values()),'initialized_equal_to_m4':True}

@contextmanager
def teacher_context(backend,runtime):
    if not getattr(backend,'joint_training',False):
        with runtime.scope(False):yield
        return
    params={canonical(n):p for n,p in backend.transformer.named_parameters() if '.lora_' in n and '.default.' in n}
    student={n:p.detach().cpu().clone() for n,p in params.items()}
    teacher=backend.fixed_m4_teacher_state
    if set(params)!=set(teacher):raise RuntimeError('Teacher layout mismatch')
    for p in backend.transformer.parameters():p.requires_grad_(False)
    with torch.no_grad():
        for n,p in params.items():p.copy_(teacher[n])
    try:
        with runtime.scope(False):yield
    finally:
        with torch.no_grad():
            for n,p in params.items():p.copy_(student[n])
        backend.set_trainable_branch('transmission')

def save_joint(path,backend,m4_sha):
    state={'sma.'+n:p for n,p in backend.sma.state_dict().items()}
    state.update({'lora.'+canonical(n):p for n,p in backend.transformer.named_parameters() if '.lora_' in n and '.default.' in n})
    atomic_safetensors(path,state,{'experiment':'SMA-B-joint','architecture':JOINT_VERSION,'sma_architecture':SMA_VERSION,'base_m4_sha256':m4_sha,'contains':'student Transmission LoRA and full SMA; no optimizer or teacher copy'})

def load_joint(path,backend,sma,device,m4_sha):
    with safe_open(path,framework='pt') as h:md=h.metadata()
    if md.get('architecture')!=JOINT_VERSION or md.get('base_m4_sha256')!=m4_sha:raise RuntimeError('Joint checkpoint identity mismatch')
    state=safetensors.torch.load_file(path,device=str(device))
    ls={n[5:]:v for n,v in state.items() if n.startswith('lora.')}
    expected={n for n,p in backend.transformer.named_parameters() if '.lora_' in n and '.default.' in n}
    if set(ls)!=expected:raise RuntimeError('Joint LoRA key mismatch; load before installing SMA')
    _,unexpected=backend.transformer.load_state_dict(ls,strict=False)
    if unexpected:raise RuntimeError(unexpected)
    sma.load_state_dict({n[4:]:v for n,v in state.items() if n.startswith('sma.')},strict=True)
    if len(state)!=len(ls)+len(sma.state_dict()):raise RuntimeError('Unexpected joint tensors')
    return md
