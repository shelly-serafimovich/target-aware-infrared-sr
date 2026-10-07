import argparse,json,os
from pathlib import Path
import numpy as np,pandas as pd
from ultralytics import YOLO

ap=argparse.ArgumentParser()
ap.add_argument('--manifest',required=True)
ap.add_argument('--run-root',required=True)
ap.add_argument('--weights',required=True)
ap.add_argument('--core',type=float,default=24.0)
ap.add_argument('--iou-match',type=float,default=0.25)
ap.add_argument('--fa-conf',type=float,default=0.25)
a=ap.parse_args()

run_root=Path(a.run_root); eval_root=run_root/'yolo_pilot_eval'; eval_root.mkdir(parents=True,exist_ok=True)
method_dirs={'BASE':run_root/'BASE','V3-A':run_root/'V3A_pooledSCR','V3-B':run_root/'V3B_pooledSCR_ND'}
model=YOLO(str(a.weights))
df=pd.read_csv(a.manifest)

def atomic_json(obj,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2),encoding='utf-8')
    os.replace(tmp,path)

def iou_one_to_many(gt,boxes):
    if len(boxes)==0: return np.zeros(0,dtype=np.float32)
    x1=np.maximum(gt[0],boxes[:,0]); y1=np.maximum(gt[1],boxes[:,1])
    x2=np.minimum(gt[2],boxes[:,2]); y2=np.minimum(gt[3],boxes[:,3])
    inter=np.maximum(0,x2-x1)*np.maximum(0,y2-y1)
    ga=max(0,gt[2]-gt[0])*max(0,gt[3]-gt[1])
    ba=np.maximum(0,boxes[:,2]-boxes[:,0])*np.maximum(0,boxes[:,3]-boxes[:,1])
    return inter/np.maximum(ga+ba-inter,1e-9)

def as_bool(x):
    if isinstance(x,(bool,np.bool_)): return bool(x)
    return str(x).strip().lower() in {'true','1','yes'}

def parse_small_gt(row):
    lp=Path(row.label_path) if isinstance(row.label_path,str) and row.label_path else None
    if lp is None or not lp.exists() or lp.stat().st_size==0: return []
    W=float(row.up_width); H=float(row.up_height); out=[]
    for line in lp.read_text().splitlines():
        q=line.strip().split()
        if len(q)<5: continue
        _,cx,cy,bw,bh=map(float,q[:5]); wp=bw*W; hp=bh*H
        if wp<=a.core and hp<=a.core:
            out.append({'x1':(cx-bw/2)*W,'y1':(cy-bh/2)*H,'x2':(cx+bw/2)*W,'y2':(cy+bh/2)*H,'w':wp,'h':hp})
    return out

for method,folder in method_dirs.items():
    sd=eval_root/method.replace(' ','_'); sd.mkdir(parents=True,exist_ok=True)
    print('\n'+'='*72); print('YOLO:',method); print('='*72)
    for j,row in enumerate(df.itertuples(index=False),1):
        stem=Path(row.image).stem; sp=sd/f'{stem}.json'
        if sp.exists(): continue
        ip=folder/row.image
        if not ip.exists(): raise FileNotFoundError(f'Missing {method} image: {ip}')
        pred=model.predict(source=str(ip),imgsz=640,conf=0.001,iou=0.7,device=0,verbose=False)[0]
        if pred.boxes is None or len(pred.boxes)==0:
            boxes=np.zeros((0,4),dtype=np.float32); conf=np.zeros(0,dtype=np.float32)
        else:
            boxes=pred.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
            conf=pred.boxes.conf.detach().cpu().numpy().astype(np.float32)
        target_rows=[]
        for ti,g in enumerate(parse_small_gt(row)):
            gv=np.array([g['x1'],g['y1'],g['x2'],g['y2']],dtype=np.float32)
            ious=iou_one_to_many(gv,boxes)
            if len(ious):
                k=int(np.argmax(ious)); best_iou=float(ious[k]); best_conf=float(conf[k])
            else: best_iou=0.; best_conf=0.
            detected=bool(best_iou>=a.iou_match)
            target_rows.append({'target_index':ti,'w':g['w'],'h':g['h'],'max_side':max(g['w'],g['h']),
                                'best_iou':best_iou,'gt_conf':best_conf if detected else 0.,'detected':detected})
        is_bg=as_bool(row.is_background)
        fa=int(np.sum(conf>=a.fa_conf)) if is_bg else 0
        atomic_json({'image':row.image,'method':method,'is_background':is_bg,'targets':target_rows,
                     'background_fa_count_conf025':fa,
                     'background_max_conf':float(conf.max()) if is_bg and len(conf) else 0.},sp)
        if j%20==0 or j==len(df): print(f'  {method}: {len(list(sd.glob("*.json")))}/{len(df)}',flush=True)

# Aggregate authoritative JSONs.
target_rows=[]; bg_rows=[]
for method in method_dirs:
    sd=eval_root/method.replace(' ','_')
    for p in sorted(sd.glob('*.json')):
        q=json.loads(p.read_text())
        if q['is_background']:
            bg_rows.append({'method':method,'image':q['image'],'fa_count':q['background_fa_count_conf025'],'max_conf':q['background_max_conf']})
        for t in q['targets']: target_rows.append({'method':method,'image':q['image'],**t})
targets=pd.DataFrame(target_rows); bgs=pd.DataFrame(bg_rows)
targets.to_csv(eval_root/'small_target_detections.csv',index=False); bgs.to_csv(eval_root/'background_false_alarms.csv',index=False)
summary=[]
for method in method_dirs:
    d=targets[targets.method==method]; b=bgs[bgs.method==method]
    summary.append({'method':method,'n_small_targets':len(d),
                    'small_target_recall_iou025':d.detected.mean() if len(d) else np.nan,
                    'mean_gt_conf':d.gt_conf.mean() if len(d) else np.nan,
                    'median_gt_conf':d.gt_conf.median() if len(d) else np.nan,
                    'n_background_images':len(b),
                    'background_images_with_FA_conf025':int((b.fa_count>0).sum()) if len(b) else 0,
                    'background_FA_count_conf025':int(b.fa_count.sum()) if len(b) else 0,
                    'background_max_conf_mean':b.max_conf.mean() if len(b) else np.nan})
summary=pd.DataFrame(summary)
base=targets[targets.method=='BASE'][['image','target_index','detected','gt_conf']].rename(columns={'detected':'base_detected','gt_conf':'base_conf'})
paired=[]
for method in ['V3-A','V3-B']:
    d=targets[targets.method==method][['image','target_index','detected','gt_conf']].rename(columns={'detected':'guided_detected','gt_conf':'guided_conf'})
    z=base.merge(d,on=['image','target_index'],how='inner')
    saved=int(((~z.base_detected)&z.guided_detected).sum()); lost=int((z.base_detected&(~z.guided_detected)).sum())
    paired.append({'method':method,'paired_targets':len(z),'saved':saved,'lost':lost,'net_saved':saved-lost,
                   'mean_conf_delta':float((z.guided_conf-z.base_conf).mean())})
paired=pd.DataFrame(paired)
base_recall=float(summary.loc[summary.method=='BASE','small_target_recall_iou025'].iloc[0])
summary['recall_delta_pp_vs_BASE']=100*(summary.small_target_recall_iou025-base_recall)
summary.to_csv(eval_root/'pilot_summary.csv',index=False); paired.to_csv(eval_root/'paired_saved_lost.csv',index=False)

geom=targets[targets.method=='BASE'][['image','target_index','max_side']].copy()
if len(geom)>=4:
    q1,q2,q3=np.quantile(geom.max_side,[.25,.5,.75]); edges=[-np.inf,q1,q2,q3,np.inf]
    labels=[f'Q1 <= {q1:.1f}px',f'Q2 <= {q2:.1f}px',f'Q3 <= {q3:.1f}px',f'Q4 > {q3:.1f}px']
    targets['size_quartile']=pd.cut(targets.max_side,bins=edges,labels=labels,include_lowest=True)
    byq=targets.groupby(['method','size_quartile'],observed=True).agg(N=('detected','size'),recall=('detected','mean'),mean_gt_conf=('gt_conf','mean')).reset_index()
    byq.to_csv(eval_root/'recall_by_small_target_quartile.csv',index=False)

print('\nPRIMARY — SMALL-TARGET RECALL (<=24x24 in pre-encoder x4 frame)')
print(summary.sort_values('small_target_recall_iou025',ascending=False).to_string(index=False))
print('\nPAIRED SAVED / LOST vs BASE')
print(paired.to_string(index=False))
print('\nSaved:',eval_root)
