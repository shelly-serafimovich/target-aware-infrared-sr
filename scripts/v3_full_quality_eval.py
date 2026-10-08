import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image
import pyiqa

ap=argparse.ArgumentParser()
ap.add_argument('--hr',required=True)
ap.add_argument('--original',required=True)
ap.add_argument('--base',required=True)
ap.add_argument('--guided',required=True)
ap.add_argument('--out',required=True)
ap.add_argument('--original-csv',default='')
a=ap.parse_args()

hr=Path(a.hr); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
method_dirs={'Original DifIISR':Path(a.original),'BASE':Path(a.base),'V3-B selected':Path(a.guided)}
exts={'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}
name_sets=[]
for d in [hr,*method_dirs.values()]:
    name_sets.append({p.name for p in d.iterdir() if p.is_file() and p.suffix.lower() in exts})
common=sorted(set.intersection(*name_sets))
print(f'Common images for quality evaluation: {len(common)}',flush=True)
(out/'common_images.txt').write_text('\n'.join(common),encoding='utf-8')

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
metric_specs=[
 ('clipiqa',{}),('musiq',{}),('niqe',{}),
 ('psnr',{'test_y_channel':True,'color_space':'ycbcr'}),
 ('lpips',{}),('ssim',{})]
metrics={}
for key,kwargs in metric_specs:
    t=time.time(); print('Loading metric:',key,flush=True)
    metrics[key]=pyiqa.create_metric(key,**kwargs).to(device)
    print(f'  ready in {time.time()-t:.1f}s',flush=True)

def load_tensor(path):
    x=np.asarray(Image.open(path).convert('RGB'),dtype=np.float32)/255.0
    return torch.from_numpy(x).permute(2,0,1).unsqueeze(0).to(device)

def run_method(method,folder):
    csv=out/f"quality_{method.replace(' ','_').replace('-','')}.csv"
    done={}
    if csv.exists():
        old=pd.read_csv(csv)
        done={str(r.image):r._asdict() for r in old.itertuples(index=False)}
        print(method,'resume rows:',len(done),flush=True)
    rows=[]
    for i,name in enumerate(common,1):
        if name in done:
            rows.append(done[name]); continue
        sr=load_tensor(folder/name); ref=load_tensor(hr/name)
        vals={'image':name,'method':method}
        with torch.no_grad():
            vals['clipiqa']=float(metrics['clipiqa'](sr).item())
            vals['musiq']=float(metrics['musiq'](sr).item())
            vals['niqe']=float(metrics['niqe'](sr).item())
            vals['psnr']=float(metrics['psnr'](sr,ref).item())
            vals['lpips']=float(metrics['lpips'](sr,ref).item())
            vals['ssim']=float(metrics['ssim'](sr,ref).item())
        rows.append(vals)
        if i%25==0 or i==len(common):
            pd.DataFrame(rows).to_csv(csv,index=False)
            print(f'{method}: {i}/{len(common)}',flush=True)
        del sr,ref
    df=pd.DataFrame(rows); df.to_csv(csv,index=False); return df

# Reuse the frozen official baseline table when it is compatible; otherwise recompute Original.
orig_df=None
if a.original_csv and Path(a.original_csv).exists():
    q=pd.read_csv(a.original_csv)
    namecol=next((c for c in ['image','filename','name','file'] if c in q.columns),None)
    need={'clipiqa','musiq','niqe','psnr','lpips','ssim'}
    if namecol and need.issubset(q.columns):
        q=q.copy(); q['image']=q[namecol].astype(str).map(lambda x:Path(x).name)
        q=q[q.image.isin(common)][['image',*sorted(need)]].copy(); q['method']='Original DifIISR'
        if len(q)==len(common):
            orig_df=q[['image','method','clipiqa','musiq','niqe','psnr','lpips','ssim']]
            orig_df.to_csv(out/'quality_Original_DifIISR.csv',index=False)
            print('Reused official Original DifIISR metrics:',len(orig_df),flush=True)
if orig_df is None:
    orig_df=run_method('Original DifIISR',method_dirs['Original DifIISR'])
base_df=run_method('BASE',method_dirs['BASE'])
guided_df=run_method('V3-B selected',method_dirs['V3-B selected'])
allq=pd.concat([orig_df,base_df,guided_df],ignore_index=True)
cols=['psnr','ssim','lpips','clipiqa','musiq','niqe']
summary=allq.groupby('method')[cols].agg(['mean','std','count'])
summary.to_csv(out/'quality_summary.csv')
allq.to_csv(out/'quality_all_methods.csv',index=False)
print('\nQUALITY SUMMARY',flush=True); print(summary,flush=True)
