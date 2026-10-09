import argparse, json, os, shutil
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image, ImageDraw
from ultralytics import YOLO

ap=argparse.ArgumentParser()
ap.add_argument('--manifest',required=True)
ap.add_argument('--original',required=True)
ap.add_argument('--base',required=True)
ap.add_argument('--guided',required=True)
ap.add_argument('--weights',required=True)
ap.add_argument('--out',required=True)
ap.add_argument('--iou-match',type=float,default=0.25)
ap.add_argument('--fa-conf',type=float,default=0.25)
a=ap.parse_args()

out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
df=pd.read_csv(a.manifest)
method_dirs={'Original DifIISR':Path(a.original),'BASE':Path(a.base),'V3-B selected':Path(a.guided)}
model=YOLO(str(a.weights))

def as_bool(x):
    if isinstance(x,(bool,np.bool_)): return bool(x)
    return str(x).strip().lower() in {'1','true','yes'}

def parse_gt(row):
    lp=Path(str(row.label_path))
    if not lp.exists() or lp.stat().st_size==0: return []
    W=float(row.up_width); H=float(row.up_height); arr=[]
    for line in lp.read_text().splitlines():
        q=line.strip().split()
        if len(q)<5: continue
        _,cx,cy,bw,bh=map(float,q[:5]); wp=bw*W; hp=bh*H
        arr.append({'x1':(cx-bw/2)*W,'y1':(cy-bh/2)*H,'x2':(cx+bw/2)*W,'y2':(cy+bh/2)*H,
                    'w':wp,'h':hp,'max_side':max(wp,hp),'small':bool(wp<=24 and hp<=24)})
    return arr

def iou_one_to_many(gt,boxes):
    if len(boxes)==0: return np.zeros(0,dtype=np.float32)
    x1=np.maximum(gt[0],boxes[:,0]); y1=np.maximum(gt[1],boxes[:,1])
    x2=np.minimum(gt[2],boxes[:,2]); y2=np.minimum(gt[3],boxes[:,3])
    inter=np.maximum(0,x2-x1)*np.maximum(0,y2-y1)
    ga=max(0,gt[2]-gt[0])*max(0,gt[3]-gt[1])
    ba=np.maximum(0,boxes[:,2]-boxes[:,0])*np.maximum(0,boxes[:,3]-boxes[:,1])
    return inter/np.maximum(ga+ba-inter,1e-9)

common=[]
for r in df.itertuples(index=False):
    if all((folder/r.image).exists() for folder in method_dirs.values()): common.append(r.image)
common=set(common); df=df[df.image.isin(common)].reset_index(drop=True)
print('Common images for detector evaluation:',len(df),flush=True)

rows=[]; bgrows=[]
for method,folder in method_dirs.items():
    cache=out/f"detections_{method.replace(' ','_').replace('-','')}.jsonl"
    done={}
    if cache.exists():
        for line in cache.read_text().splitlines():
            if line.strip():
                q=json.loads(line); done[q['image']]=q
    print('\n',method,'resume',len(done),flush=True)
    records=[]
    for i,row in enumerate(df.itertuples(index=False),1):
        if row.image in done:
            rec=done[row.image]
        else:
            pred=model.predict(source=str(folder/row.image),imgsz=640,conf=0.001,iou=0.7,device=0,verbose=False)[0]
            if pred.boxes is None or len(pred.boxes)==0:
                boxes=np.zeros((0,4),np.float32); conf=np.zeros(0,np.float32)
            else:
                boxes=pred.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
                conf=pred.boxes.conf.detach().cpu().numpy().astype(np.float32)
            targets=[]
            for ti,g in enumerate(parse_gt(row)):
                gv=np.array([g['x1'],g['y1'],g['x2'],g['y2']],dtype=np.float32)
                ious=iou_one_to_many(gv,boxes)
                if len(ious):
                    k=int(np.argmax(ious)); biou=float(ious[k]); bconf=float(conf[k])
                else: biou=0.; bconf=0.
                det=bool(biou>=a.iou_match)
                targets.append({'target_index':ti,**g,'best_iou':biou,'gt_conf':bconf if det else 0.,'detected':det})
            isbg=as_bool(row.is_background)
            rec={'image':row.image,'is_background':isbg,'targets':targets,
                 'fa_count':int(np.sum(conf>=a.fa_conf)) if isbg else 0,
                 'max_conf':float(conf.max()) if isbg and len(conf) else 0.}
            with open(cache,'a',encoding='utf-8') as f: f.write(json.dumps(rec)+'\n')
        records.append(rec)
        if i%50==0 or i==len(df): print(f'{method}: {i}/{len(df)}',flush=True)
    for rec in records:
        if rec['is_background']:
            bgrows.append({'method':method,'image':rec['image'],'fa_count':rec['fa_count'],'max_conf':rec['max_conf']})
        for t in rec['targets']:
            rows.append({'method':method,'image':rec['image'],**t})

targets=pd.DataFrame(rows); bgs=pd.DataFrame(bgrows)
targets.to_csv(out/'target_level_detections.csv',index=False); bgs.to_csv(out/'background_false_alarms.csv',index=False)
summary=[]
for method in method_dirs:
    d=targets[targets.method==method]; s=d[d.small==True]; b=bgs[bgs.method==method]
    summary.append({'method':method,'n_images':len(df),'n_targets':len(d),'target_recall_iou025':d.detected.mean(),
                    'mean_gt_conf_all':d.gt_conf.mean(),'n_small_targets':len(s),'small_target_recall_iou025':s.detected.mean() if len(s) else np.nan,
                    'mean_gt_conf_small':s.gt_conf.mean() if len(s) else np.nan,'n_background_images':len(b),
                    'background_images_with_FA_conf025':int((b.fa_count>0).sum()),'background_FA_count_conf025':int(b.fa_count.sum()),
                    'background_max_conf_mean':b.max_conf.mean()})
summary=pd.DataFrame(summary)

# Paired saved/lost relative to BASE, both all-target and <=24x24 target regimes.
base=targets[targets.method=='BASE'][['image','target_index','small','detected','gt_conf']].rename(columns={'detected':'base_detected','gt_conf':'base_conf'})
paired=[]
for method in ['Original DifIISR','V3-B selected']:
    d=targets[targets.method==method][['image','target_index','detected','gt_conf']].rename(columns={'detected':'other_detected','gt_conf':'other_conf'})
    z=base.merge(d,on=['image','target_index'],how='inner')
    for scope,zz in [('all',z),('small<=24',z[z.small==True])]:
        saved=int(((~zz.base_detected)&zz.other_detected).sum()); lost=int((zz.base_detected&(~zz.other_detected)).sum())
        paired.append({'method':method,'scope':scope,'paired_targets':len(zz),'saved':saved,'lost':lost,'net_saved':saved-lost,
                       'mean_conf_delta':float((zz.other_conf-zz.base_conf).mean())})
pd.DataFrame(paired).to_csv(out/'paired_saved_lost.csv',index=False)

# Save every disagreement as a project-ready visual triptych.
visual_root=out/'visual_examples'; visual_root.mkdir(parents=True,exist_ok=True)
index_rows=[]

def panel(path,title,det,conf,box):
    im=Image.open(path).convert('RGB')
    x1,y1,x2,y2=[int(round(v)) for v in box]
    dr=ImageDraw.Draw(im); dr.rectangle([x1,y1,x2,y2],outline=(255,0,0),width=2)
    band=Image.new('RGB',(im.width,30),'white'); bd=ImageDraw.Draw(band)
    bd.text((5,7),f'{title} | detected={"YES" if det else "NO"} | conf={conf:.3f}',fill='black')
    outp=Image.new('RGB',(im.width,im.height+30),'white'); outp.paste(band,(0,0)); outp.paste(im,(0,30))
    return outp

def save_case(row,category):
    cdir=visual_root/category; cdir.mkdir(parents=True,exist_ok=True)
    box=(row.x1,row.y1,row.x2,row.y2)
    p_orig=panel(method_dirs['Original DifIISR']/row.image,'Original DifIISR',bool(row.original_detected),float(row.original_conf),box)
    p_base=panel(method_dirs['BASE']/row.image,'BASE',bool(row.base_detected),float(row.base_conf),box)
    p_v3=panel(method_dirs['V3-B selected']/row.image,'V3-B selected',bool(row.v3b_detected),float(row.v3b_conf),box)
    H=max(p_orig.height,p_base.height,p_v3.height); W=p_orig.width+p_base.width+p_v3.width
    canvas=Image.new('RGB',(W,H),'white'); x=0
    for p in [p_orig,p_base,p_v3]: canvas.paste(p,(x,0)); x+=p.width
    fp=cdir/f'{Path(row.image).stem}_t{int(row.target_index)}.png'; canvas.save(fp)
    index_rows.append({'category':category,'image':row.image,'target_index':int(row.target_index),
                       'small':bool(row.small),'base_detected':bool(row.base_detected),'base_conf':float(row.base_conf),
                       'original_detected':bool(row.original_detected),'original_conf':float(row.original_conf),
                       'v3b_detected':bool(row.v3b_detected),'v3b_conf':float(row.v3b_conf),'visual_path':str(fp)})

geom=targets[targets.method=='BASE'][['image','target_index','x1','y1','x2','y2','small']]
bdet=targets[targets.method=='BASE'][['image','target_index','detected','gt_conf']].rename(columns={'detected':'base_detected','gt_conf':'base_conf'})
odet=targets[targets.method=='Original DifIISR'][['image','target_index','detected','gt_conf']].rename(columns={'detected':'original_detected','gt_conf':'original_conf'})
vdet=targets[targets.method=='V3-B selected'][['image','target_index','detected','gt_conf']].rename(columns={'detected':'v3b_detected','gt_conf':'v3b_conf'})
cmp=geom.merge(bdet,on=['image','target_index']).merge(odet,on=['image','target_index']).merge(vdet,on=['image','target_index'])

case_masks={
    'V3B_saved_vs_BASE':(~cmp.base_detected)&cmp.v3b_detected,
    'V3B_lost_vs_BASE':cmp.base_detected&(~cmp.v3b_detected),
    'V3B_saved_vs_Original':(~cmp.original_detected)&cmp.v3b_detected,
    'V3B_lost_vs_Original':cmp.original_detected&(~cmp.v3b_detected),
}
for category,mask in case_masks.items():
    subset=cmp[mask]
    print(f'{category}: {len(subset)} differing targets',flush=True)
    for row in subset.itertuples(index=False): save_case(row,category)

pd.DataFrame(index_rows).to_csv(visual_root/'visual_example_index.csv',index=False)
print('Visual examples saved to:',visual_root,flush=True)

# Standard Ultralytics validation: precision / recall / AP50 / mAP50-95.
# Build the temporary dataset locally. The detector is single-class (drone=0), while
# the corrected annotation files can retain source-dataset class IDs (e.g. 2 or 4).
# For the standard one-class detector evaluation, remap every non-empty GT line to
# class 0 while preserving its normalized bbox coordinates.
val_rows=[]
local_val_root=Path('/content/v3b_yolo_dataset_cache')
for method,folder in method_dirs.items():
    root=local_val_root/method.replace(' ','_').replace('-','')
    if root.exists(): shutil.rmtree(root)
    imdir=root/'images'/'val'; lbdir=root/'labels'/'val'; imdir.mkdir(parents=True,exist_ok=True); lbdir.mkdir(parents=True,exist_ok=True)
    for row in df.itertuples(index=False):
        src=(folder/row.image).resolve(); dst=imdir/row.image
        os.symlink(str(src),str(dst))
        lp=Path(str(row.label_path)); ldst=lbdir/(Path(row.image).stem+'.txt')
        if lp.exists() and lp.stat().st_size>0:
            remapped=[]
            for line in lp.read_text().splitlines():
                q=line.strip().split()
                if len(q)>=5:
                    remapped.append('0 ' + ' '.join(q[1:5]))
            ldst.write_text(('\n'.join(remapped)+'\n') if remapped else '')
        else:
            ldst.write_text('')
    yaml=root/'data.yaml'; yaml.write_text(f"path: {root}\ntrain: images/val\nval: images/val\nnames:\n  0: drone\n")
    print(f'Running standard YOLO val for {method} ...',flush=True)
    val=model.val(data=str(yaml),split='val',imgsz=640,device=0,verbose=False,plots=False,save_json=False)
    val_rows.append({'method':method,'precision':float(val.box.mp),'recall':float(val.box.mr),'AP50':float(val.box.map50),'mAP50_95':float(val.box.map)})
valdf=pd.DataFrame(val_rows); valdf.to_csv(out/'yolo_dataset_summary.csv',index=False)
summary=summary.merge(valdf,on='method',how='left')
summary.to_csv(out/'detection_summary.csv',index=False)
print('\nDETECTION SUMMARY'); print(summary.to_string(index=False))
print('\nPAIRED SAVED / LOST'); print(pd.DataFrame(paired).to_string(index=False))
