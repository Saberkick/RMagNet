"""M5-R: bounded signed multiband RGB correction; no generative backbone."""
import torch
from torch import nn
import torch.nn.functional as F

VERSION='m5-c1-pixel-v1'

class LayerNorm2d(nn.Module):
    def __init__(self,c):
        super().__init__();self.weight=nn.Parameter(torch.ones(1,c,1,1));self.bias=nn.Parameter(torch.zeros(1,c,1,1))
    def forward(self,x):
        mean=x.mean(1,keepdim=True);var=(x-mean).square().mean(1,keepdim=True)
        return (x-mean)*torch.rsqrt(var+1e-6)*self.weight+self.bias

class NAFBlock(nn.Module):
    def __init__(self,c):
        super().__init__();self.norm1=LayerNorm2d(c);self.norm2=LayerNorm2d(c)
        self.expand=nn.Conv2d(c,2*c,1);self.dw=nn.Conv2d(2*c,2*c,3,padding=1,groups=2*c)
        self.attn=nn.Conv2d(c,c,1);self.project=nn.Conv2d(c,c,1)
        self.ff1=nn.Conv2d(c,2*c,1);self.ff2=nn.Conv2d(c,c,1)
        self.beta=nn.Parameter(torch.zeros(1,c,1,1));self.gamma=nn.Parameter(torch.zeros(1,c,1,1))
    def forward(self,x):
        a,b=self.dw(self.expand(self.norm1(x))).chunk(2,1);v=a*b
        v=v*self.attn(F.adaptive_avg_pool2d(v,1));y=x+self.beta*self.project(v)
        a,b=self.ff1(self.norm2(y)).chunk(2,1)
        return y+self.gamma*self.ff2(a*b)

def down(x):
    k=x.new_tensor([1,4,6,4,1])/16;k=(k[:,None]*k[None,:]).expand(x.shape[1],1,5,5)
    return F.conv2d(F.pad(x,(2,2,2,2),mode='reflect'),k,stride=2,groups=x.shape[1])

def up(x,size):return F.interpolate(x,size=size,mode='bilinear',align_corners=False)

def pyramid(x):
    g=[x]
    for _ in range(3):g.append(down(g[-1]))
    return g,[g[l]-up(g[l+1],g[l].shape[-2:]) for l in range(3)]

class PixelRefiner(nn.Module):
    def __init__(self,width=32):
        super().__init__();self.width=width
        self.stems=nn.ModuleList([nn.Conv2d(9,width,3,padding=1) for _ in range(4)])
        self.blocks=nn.ModuleList([nn.Sequential(NAFBlock(width),NAFBlock(width)) for _ in range(4)])
        self.context=nn.Conv2d(width,width,1)
        self.fuse=nn.ModuleList([nn.Conv2d(2*width,width,1) for _ in range(3)])
        self.heads=nn.ModuleList([nn.Conv2d(width,3,3,padding=1) for _ in range(4)])
        for head in self.heads:nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
        self.limits=(.03,.05,.08,.12)
    def forward(self,image,t0):
        if image.shape!=t0.shape or image.ndim!=4:raise ValueError('Aligned BCHW RGB required')
        gi,_=pyramid(image);gt,_=pyramid(t0)
        f=[self.stems[l](torch.cat((gi[l],gt[l],gi[l]-gt[l]),1)) for l in range(4)]
        f[3]=self.blocks[3](f[3]);context=self.context(F.adaptive_avg_pool2d(f[3],1))
        for l in (2,1,0):f[l]=self.blocks[l](self.fuse[l](torch.cat((f[l],up(f[l+1],f[l].shape[-2:])),1))+context)
        bands=[self.limits[l]*torch.tanh(self.heads[l](f[l])) for l in range(4)]
        residual=bands[3]
        for l in (2,1,0):residual=up(residual,bands[l].shape[-2:])+bands[l]
        delta=.20*torch.tanh(residual/.20)
        return {'prediction':(t0+delta).clamp(0,1),'delta':delta,'bands':bands}

def ssim(x,y):
    x=x.float();y=y.float();mx=F.avg_pool2d(x,11,1,5);my=F.avg_pool2d(y,11,1,5)
    vx=F.avg_pool2d(x*x,11,1,5)-mx.square();vy=F.avg_pool2d(y*y,11,1,5)-my.square()
    cov=F.avg_pool2d(x*y,11,1,5)-mx*my
    return (((2*mx*my+.01**2)*(2*cov+.03**2))/((mx.square()+my.square()+.01**2)*(vx+vy+.03**2))).mean()

def edge(x,y):
    return .5*(F.l1_loss(x[...,1:,:]-x[...,:-1,:],y[...,1:,:]-y[...,:-1,:])+F.l1_loss(x[...,:,1:]-x[...,:,:-1],y[...,:,1:]-y[...,:,:-1]))

def losses(result,t0,gt):
    pred=result['prediction'].float();gt=gt.float();t0=t0.float()
    gg,hg=pyramid(gt);g0,h0=pyramid(t0)
    targets=[hg[l]-h0[l] for l in range(3)]+[gg[3]-g0[3]]
    band=sum(F.l1_loss(b.float(),v) for b,v in zip(result['bands'],targets))/4
    mask=((t0-gt).abs().mean(1,keepdim=True)<.02).float()
    keep=(result['delta'].abs()*mask).sum()/(3*mask.sum()).clamp_min(1)
    terms={'charb':((pred-gt).square()+1e-6).sqrt().mean(),'ssim':1-ssim(pred,gt),'edge':edge(pred,gt),'band':band,'keep':keep}
    total=terms['charb']+.2*terms['ssim']+.1*terms['edge']+.1*terms['band']+.05*terms['keep']
    return total,terms
