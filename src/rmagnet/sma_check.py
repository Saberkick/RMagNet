"""Small GPU checks for identity init, adapter gradients and scoped recomputation."""
import json,tempfile
from pathlib import Path
import torch
from torch import nn
import safetensors.torch
from .sma import SMA,Runtime,MemoryWrappedBlock


class Dummy(nn.Module):
    def forward(self,image,text,*args,**kwargs):return text,image*1.01


def main():
    torch.manual_seed(2026);torch.set_num_threads(1)
    s=SMA().cuda();s.freeze_memory()
    with torch.no_grad():s.projection.basis[:256].copy_(torch.eye(256,device='cuda'))
    rt=Runtime(s)
    wrappers=[MemoryWrappedBlock(Dummy(),rt,n) for n in (37,39,41)]
    h=torch.randn(1,24,3072,device='cuda',dtype=torch.bfloat16)
    text=torch.zeros(1,1,8,device='cuda',dtype=torch.bfloat16)
    with rt.scope(True,(4,6)):
        x=h
        for block in wrappers:text,x=rt.checkpoint(block,x,text)
    expected=h
    for _ in wrappers:expected=expected*1.01
    assert torch.equal(x,expected),'Zero output heads changed baseline'
    # Deliberately run a teacher forward before generator backward: old graph
    # recomputation must use its captured enabled state and detached memory.
    leaf=x.detach().requires_grad_(True)
    with rt.scope(False):
        t=leaf
        for block in wrappers:_,t=rt.checkpoint(block,t,text)
        teacher_grad=torch.autograd.grad(t.float().square().mean(),leaf)[0]
    x.backward(teacher_grad)
    parameters=list(s.readers.parameters())
    norm=torch.nn.utils.clip_grad_norm_(parameters,1.0)
    assert torch.isfinite(norm) and norm>0,'No reader gradient'
    assert all(p.grad is None for p in s.memory.parameters()),'Memory received gradients'
    before=s.readers['39'].up.weight.detach().clone()
    opt=torch.optim.AdamW(parameters,lr=1e-4);opt.step()
    assert not torch.equal(before,s.readers['39'].up.weight),'No parameter update'
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'sma.safetensors';safetensors.torch.save_file({k:v.contiguous() for k,v in s.state_dict().items()},p)
        copy=SMA().cuda();copy.load_state_dict(safetensors.torch.load_file(p,device='cuda'),strict=True)
        assert all(torch.equal(v,copy.state_dict()[k]) for k,v in s.state_dict().items())
    # Extreme-aspect shapes and spatially displaced memory remain finite.
    for grid in ((15,52),(44,18)):
        z=torch.randn(1,grid[0]*grid[1],256,device='cuda')
        with torch.no_grad():m=s.memory(z,grid)
        assert m.shape==z.shape and torch.isfinite(m).all()
    print(json.dumps({'status':'module_checks_passed','identity_init':True,'checkpoint_teacher_isolation':True,'nonzero_reader_gradient':float(norm),'frozen_memory':True,'checkpoint_roundtrip':True,'extreme_aspect_grids':True,'reader_parameters':sum(p.numel() for p in parameters)}),flush=True)


if __name__=='__main__':main()
