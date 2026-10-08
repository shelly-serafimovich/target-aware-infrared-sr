import argparse,sys,random,time,gc,json,hashlib,os
# DifIISR sampling asserts that exactly one GPU is visible.
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
# Reduce CUDA allocator fragmentation on 16 GB-class Colab GPUs.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','max_split_size_mb:128')
from pathlib import Path
import numpy as np,pandas as pd,torch
import torch.nn.functional as F
from PIL import Image
from omegaconf import OmegaConf
from torch.utils.checkpoint import checkpoint

sys.path.insert(0,'/content/DifIISR')
import ldm.modules.diffusionmodules.model as ae_model
ae_model.XFORMERS_IS_AVAILBLE=False
from sampler import DifIISRSampler

ap=argparse.ArgumentParser()
ap.add_argument('--manifest',required=True)
ap.add_argument('--lr',required=True)
ap.add_argument('--out',required=True)
ap.add_argument('--config',required=True)
ap.add_argument('--difiisr-config',required=True)
a=ap.parse_args()

cfgj=json.loads(Path(a.config).read_text())
SEED=int(cfgj['seed'])
CORE=int(cfgj['core']); OUTER=int(cfgj['outer']); TOP_K=int(cfgj['top_k'])
SIGMA_FLOOR=float(cfgj['sigma_floor']); BLOB_K=float(cfgj['blob_k'])
SOFT_T=float(cfgj['soft_t']); SOFT_MIN_SUPPORT=float(cfgj['soft_min_support'])
SCR_GATE=float(cfgj['scr_gate']); SCR_GATE_TEMP=float(cfgj['scr_gate_temp'])
ND_GATE=float(cfgj['nd_gate']); ND_GATE_TEMP=float(cfgj['nd_gate_temp'])
ND_FLOOR=float(cfgj['nd_floor_weight'])
MAX_CAND=int(cfgj['max_candidates']); PROPOSAL_NMS=int(cfgj['proposal_nms'])
SCR_GOAL=float(cfgj['scr_goal']); GOAL_TEMP=float(cfgj['goal_temp'])
STEPS=int(cfgj['guidance_steps']); LR_GUIDE=float(cfgj['guidance_lr'])
LAMBDA_FID=float(cfgj['lambda_fid']); LAMBDA_BG=float(cfgj['lambda_bg'])
# Pilot keeps V3-A for the ablation. Full validation only needs BASE + selected V3-B.
RUN_V3A=bool(cfgj.get('run_v3a', 'full' not in str(a.manifest).lower()))

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

out=Path(a.out); lrdir=Path(a.lr)
for d in ['BASE','V3A_pooledSCR','V3B_pooledSCR_ND','status']:
    (out/d).mkdir(parents=True,exist_ok=True)
log_path=out/'run.log'

def log(msg):
    s=str(msg)
    print(s,flush=True)
    with open(log_path,'a',encoding='utf-8') as f:
        f.write(s+'\n'); f.flush()

def atomic_json(obj,path):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2),encoding='utf-8')
    os.replace(tmp,path)

def atomic_image(x,path,h,w):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    x=x[...,:h*4,:w*4]
    im=((x[0].float().mean(0).detach().cpu().numpy()+1.)*.5*255.).clip(0,255).round().astype(np.uint8)
    tmp=path.with_name(path.stem+'.tmp'+path.suffix)
    Image.fromarray(im).save(tmp)
    os.replace(tmp,path)

def image_seed(name):
    h=int(hashlib.sha1(name.encode('utf-8')).hexdigest()[:8],16)
    return SEED + (h % 100000000)

cfg=OmegaConf.load(a.difiisr_config)
cfg.model.ckpt_path='/content/DifIISR/weights/DifIISR.pth'
cfg.diffusion.params.sf=4
cfg.autoencoder.ckpt_path='/content/DifIISR/weights/autoencoder_vq_f4.pth'
s=DifIISRSampler(cfg,chop_size=512,chop_stride=448,chop_bs=1,use_fp16=True,seed=SEED,ddim=True)

for module in [s.model,s.autoencoder]:
    for p in module.parameters(): p.requires_grad_(False)

# The differentiable VQ decoder is the memory-heavy part of guidance. Checkpoint
# it so activations are recomputed during backward instead of being stored.
_orig_decoder_forward=s.autoencoder.decoder.forward
def _checkpointed_decoder_forward(*args, **kwargs):
    if torch.is_grad_enabled() and any(torch.is_tensor(x) and x.requires_grad for x in args):
        return checkpoint(lambda *xs: _orig_decoder_forward(*xs, **kwargs), *args, use_reentrant=False)
    return _orig_decoder_forward(*args, **kwargs)
s.autoencoder.decoder.forward=_checkpointed_decoder_forward

def pad64(y):
    h,w=y.shape[-2:]; ph=(64-h%64)%64; pw=(64-w%64)%64
    return F.pad(y,(0,pw,0,ph),mode='reflect'),h,w

@torch.no_grad()
def propose_centers(y_up,crop_h,crop_w):
    g=y_up.mean(1,keepdim=True).float()
    k=49
    mu=F.avg_pool2d(g,k,1,k//2)
    mu2=F.avg_pool2d(g*g,k,1,k//2)
    std=torch.sqrt((mu2-mu*mu).clamp_min(0.)+1e-6)
    z=(g-mu)/std.clamp_min(SIGMA_FLOOR)
    mx=F.max_pool2d(z,PROPOSAL_NMS,1,PROPOSAL_NMS//2)
    score=z.clone()
    valid=(z>=mx-1e-7)
    spatial=torch.zeros_like(valid)
    half=OUTER//2
    y1=max(half,crop_h-half); x1=max(half,crop_w-half)
    spatial[...,half:y1,half:x1]=True
    score=score.masked_fill(~(valid & spatial),-1e9)
    flat=score.flatten()
    n=min(MAX_CAND,int((score>-1e8).sum().item()))
    if n<=0: return []
    inds=torch.topk(flat,n).indices
    W=score.shape[-1]
    return [(int(i//W),int(i%W)) for i in inds]

def patch_metrics(gray,centers):
    scr=[]; nd=[]
    half=OUTER//2
    c0=(OUTER-CORE)//2; c1=c0+CORE
    for cy,cx in centers:
        outer=gray[...,cy-half:cy+half,cx-half:cx+half]
        core=outer[...,c0:c1,c0:c1]
        top=outer[...,:c0,:].flatten()
        bottom=outer[...,c1:,:].flatten()
        left=outer[...,c0:c1,:c0].flatten()
        right=outer[...,c0:c1,c1:].flatten()
        ring=torch.cat([top,bottom,left,right])
        ring_mu=ring.mean()
        ring_std=ring.std(unbiased=False).clamp_min(SIGMA_FLOOR)
        core_flat=core.flatten()
        pool=torch.topk(core_flat,min(TOP_K,core_flat.numel())).values.mean()
        sc=(pool-ring_mu)/ring_std
        z=(core-ring_mu)/ring_std
        m=torch.sigmoid((z-BLOB_K)/SOFT_T)
        total=m.sum()
        neigh=(m*F.avg_pool2d(m,3,1,1)).sum()
        dens=neigh/torch.clamp(total,min=SOFT_MIN_SUPPORT)
        scr.append(sc); nd.append(dens)
    if not scr:
        dev=gray.device
        return torch.zeros(0,device=dev),torch.zeros(0,device=dev)
    return torch.stack(scr),torch.stack(nd)

def frozen_weights(scr_in,nd_in,variant):
    ws=torch.sigmoid((scr_in-SCR_GATE)/SCR_GATE_TEMP)
    if variant=='A':
        return ws.detach()
    wnd=torch.sigmoid((nd_in-ND_GATE)/ND_GATE_TEMP)
    return (ws*(ND_FLOOR+(1.-ND_FLOOR)*wnd)).detach()

def influence_mask(shape,centers,weights):
    m=torch.zeros((1,1,shape[-2],shape[-1]),device=weights.device,dtype=torch.float32)
    half=OUTER//2
    for (cy,cx),w in zip(centers,weights):
        yy=slice(cy-half,cy+half); xx=slice(cx-half,cx+half)
        m[...,yy,xx]=torch.maximum(m[...,yy,xx],w.float())
    return m.detach().clamp(0,1)

def weighted_mean(x,w):
    return (x*w).sum()/w.sum().clamp_min(1e-6)

def guide(z0,base,centers,w):
    zv=z0.detach().float().clone().requires_grad_(True)
    opt=torch.optim.Adam([zv],lr=LR_GUIDE)
    influence=influence_mask(base.shape,centers,w)
    bgmask=(1.-influence)
    trace=[]
    for step in range(STEPS):
        opt.zero_grad(set_to_none=True)
        dec=s.base_diffusion.decode_first_stage(zv,s.autoencoder,no_grad=False).float().clamp(-1,1)
        gray=dec.mean(1,keepdim=True)
        scr_cur,nd_cur=patch_metrics(gray,centers)
        deficit=F.softplus((SCR_GOAL-scr_cur)/GOAL_TEMP)*GOAL_TEMP
        target=weighted_mean(deficit,w)
        diff=(dec-base.float())
        fid=(diff*diff).mean()
        bg=((diff*diff)*bgmask).sum()/(bgmask.sum().clamp_min(1.)*diff.shape[1])
        loss=target+LAMBDA_FID*fid+LAMBDA_BG*bg
        loss.backward(); opt.step()
        trace.append({'step':step+1,'loss':float(loss.detach()),'target':float(target.detach()),
                      'fid':float(fid.detach()),'bg_fid':float(bg.detach()),
                      'scr_wmean':float(weighted_mean(scr_cur.detach(),w)),
                      'nd_wmean':float(weighted_mean(nd_cur.detach(),w))})
        del dec,gray,scr_cur,nd_cur,deficit,target,diff,fid,bg,loss
        torch.cuda.empty_cache()
    with torch.no_grad():
        dec=s.base_diffusion.decode_first_stage(zv,s.autoencoder).float().clamp(-1,1)
        scr_end,nd_end=patch_metrics(dec.mean(1,keepdim=True),centers)
        diff=dec-base.float()
        mse=float((diff*diff).mean())
    return dec.detach(),trace,scr_end.detach(),nd_end.detach(),mse

df=pd.read_csv(a.manifest)
log('='*88)
log(f'V3 recall-first run | images={len(df)} | steps={STEPS} | lr={LR_GUIDE} | goal={SCR_GOAL} | V3A={RUN_V3A}')
log(f'core/outer={CORE}/{OUTER} | topK={TOP_K} | SCR gate={SCR_GATE} | ND gate={ND_GATE}')
log('='*88)
start=time.time()

for idx,r in df.iterrows():
    name=r.image; stem=Path(name).stem
    status_path=out/'status'/f'{stem}.json'
    paths={'BASE':out/'BASE'/name,
           'V3A':out/'V3A_pooledSCR'/name,
           'V3B':out/'V3B_pooledSCR_ND'/name}
    required=[paths['BASE'],paths['V3B']] + ([paths['V3A']] if RUN_V3A else [])
    if status_path.exists() and all(p.exists() for p in required):
        if (idx+1)%20==0 or idx==0:
            log(f'[{idx+1}/{len(df)}] SKIP complete {name}')
        continue

    t0=time.time()
    seed_i=image_seed(name)
    arr=np.asarray(Image.open(lrdir/name).convert('RGB'),dtype=np.float32)/255.
    y=torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).cuda()
    y=(y-.5)/.5
    y,h,w=pad64(y)
    crop_h,crop_w=h*4,w*4

    with torch.no_grad():
        y_up=F.interpolate(y,scale_factor=4,mode='bicubic')
        centers=propose_centers(y_up,crop_h,crop_w)
        if not centers:
            centers=[(crop_h//2,crop_w//2)]
        scr_in,nd_in=patch_metrics(y_up.mean(1,keepdim=True).float(),centers)
        wA=frozen_weights(scr_in,nd_in,'A') if RUN_V3A else None
        wB=frozen_weights(scr_in,nd_in,'B')

        model_kwargs={'lq':y} if s.configs.model.params.cond_lq else None
        z_y=s.base_diffusion.encode_first_stage(y,s.autoencoder,up_sample=True)
        gen=torch.Generator(device=y.device); gen.manual_seed(seed_i)
        noise=torch.randn(z_y.shape,generator=gen,device=z_y.device,dtype=z_y.dtype)
        z=s.base_diffusion.prior_sample(z_y,noise)
        i=s.base_diffusion.num_timesteps-1
        t=torch.tensor([i],device=y.device)
        o=s.base_diffusion.ddim_sample(model=s.model,x=z,y=z_y,t=t,
                                      clip_denoised=False,model_kwargs=model_kwargs)
        z0=o['pred_xstart'].detach()
        base=s.base_diffusion.decode_first_stage(z0,s.autoencoder).detach().float().clamp(-1,1)

    atomic_image(base,paths['BASE'],h,w)

    # Free diffusion-only tensors before any differentiable decoder pass.
    del z_y,noise,z,t,o
    gc.collect(); torch.cuda.empty_cache()

    if RUN_V3A:
        a_img,a_trace,a_scr,a_nd,a_mse=guide(z0,base,centers,wA)
        atomic_image(a_img,paths['V3A'],h,w)
        del a_img; gc.collect(); torch.cuda.empty_cache()
    else:
        a_trace=[]; a_scr=torch.zeros(0); a_nd=torch.zeros(0); a_mse=float('nan')

    b_img,b_trace,b_scr,b_nd,b_mse=guide(z0,base,centers,wB)
    atomic_image(b_img,paths['V3B'],h,w)
    del b_img

    status={
        'image':name,'pilot_group':str(r.pilot_group),'seed':int(seed_i),
        'lr_shape':[int(h),int(w)],'crop_shape_x4':[int(crop_h),int(crop_w)],
        'n_candidates':len(centers),'centers_yx':centers,
        'input_scr':[float(x) for x in scr_in.detach().cpu()],
        'input_neighbour_density':[float(x) for x in nd_in.detach().cpu()],
        'weight_v3a':[float(x) for x in wA.detach().cpu()] if wA is not None else [],
        'weight_v3b':[float(x) for x in wB.detach().cpu()],
        'v3a_trace':a_trace,'v3b_trace':b_trace,
        'v3a_final_scr':[float(x) for x in a_scr.cpu()],
        'v3a_final_nd':[float(x) for x in a_nd.cpu()],
        'v3b_final_scr':[float(x) for x in b_scr.cpu()],
        'v3b_final_nd':[float(x) for x in b_nd.cpu()],
        'v3a_mse_to_base':a_mse,'v3b_mse_to_base':b_mse,
        'elapsed_sec':time.time()-t0
    }
    atomic_json(status,status_path)

    elapsed=time.time()-start
    done=len(list((out/'status').glob('*.json')))
    rate=done/elapsed if elapsed else 0
    eta=(len(df)-done)/rate/60 if rate else float('nan')
    if RUN_V3A:
        guide_msg=f'SCRend A/B={a_trace[-1]["scr_wmean"]:.2f}/{b_trace[-1]["scr_wmean"]:.2f} | MSE A/B={a_mse:.5f}/{b_mse:.5f}'
    else:
        guide_msg=f'SCRend B={b_trace[-1]["scr_wmean"]:.2f} | MSE B={b_mse:.5f}'
    log(f'[{idx+1}/{len(df)}] DONE {name} | cand={len(centers)} | '
        f'SCRin max={float(scr_in.max()):.2f} | ND max={float(nd_in.max()):.2f} | '
        f'{guide_msg} | {time.time()-t0:.1f}s | ETA {eta:.1f}m')

    del y,y_up,z0,base,scr_in,nd_in,wA,wB,a_scr,a_nd,b_scr,b_nd
    gc.collect(); torch.cuda.empty_cache()

rows=[]
for p in sorted((out/'status').glob('*.json')):
    q=json.loads(p.read_text())
    def last(trace,key):
        return trace[-1][key] if trace else np.nan
    rows.append({
        'image':q['image'],'pilot_group':q['pilot_group'],'n_candidates':q['n_candidates'],
        'input_scr_max':max(q['input_scr']) if q['input_scr'] else np.nan,
        'input_nd_max':max(q['input_neighbour_density']) if q['input_neighbour_density'] else np.nan,
        'v3a_scr_wmean_end':last(q['v3a_trace'],'scr_wmean'),
        'v3b_scr_wmean_end':last(q['v3b_trace'],'scr_wmean'),
        'v3a_mse_to_base':q['v3a_mse_to_base'],'v3b_mse_to_base':q['v3b_mse_to_base'],
        'elapsed_sec':q['elapsed_sec']
    })
pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
log(f'✓ COMPLETE statuses={len(rows)}/{len(df)} | summary={out/"metrics.csv"}')
