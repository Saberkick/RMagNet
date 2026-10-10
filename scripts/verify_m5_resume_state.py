"""CPU synthetic next-update equivalence, no experiment metric or C1 prediction."""
import json,random,sys
from pathlib import Path
import torch
from src.rmagnet import resume_bundle as bundle
from src.rmagnet.m5_pixel import PixelRefiner,losses
from src.rmagnet.m5_train import save_model,load_model

root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(1);torch.manual_seed(2026);random.seed(2026)
image=torch.rand(1,3,32,48);gt=torch.rand_like(image)
cache={'checkpoint_sha256':'0'*64,'dataset_manifest_sha256':'1'*64}
identity={'test':'synthetic M5 optimizer/RNG roundtrip only'}
model=PixelRefiner();optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:(s+1)/51)
def update(m,opt,sched):
    opt.zero_grad(set_to_none=True)
    i,t=(image.flip(-1),gt.flip(-1)) if random.random()<.5 else (image,gt)
    value,_=losses(m(i,i),i,t);value.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();sched.step()
    return float(value.detach())
update(model,optimizer,scheduler)
cursor={'epoch':0,'batch':1,'step':1}
state={'cursor':cursor,'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'rng':bundle.rng_state()}
bundle.save(root,'latest',lambda p:save_model(p,model,cache,0,1),state,identity,cursor)
reference_loss=update(model,optimizer,scheduler)
reference={k:v.detach().clone() for k,v in model.state_dict().items()}
ck,saved=bundle.load(root,identity)
restored,_=load_model(ck/'model.safetensors',cache,torch.device('cpu'))
opt=torch.optim.AdamW(restored.parameters(),lr=2e-4,weight_decay=1e-4)
sc=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:(s+1)/51)
opt.load_state_dict(saved['optimizer']);sc.load_state_dict(saved['scheduler']);bundle.restore_rng(saved['rng'])
restored_loss=update(restored,opt,sc)
maximum=max(float((reference[k]-v).abs().max()) for k,v in restored.state_dict().items())
assert maximum==0 and reference_loss==restored_loss
result={'status':'passed','synthetic_test_only':True,'next_update_max_parameter_difference':maximum,
        'next_update_loss_equal':True,'optimizer_state_count':len(opt.state),'rng_restored':True}
(root/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
