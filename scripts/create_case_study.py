from PIL import Image
import os
fig_dir='essay/figures'
imgs=['sample_221.png','sample_797.png','vis_prediction_399.png','vis_prediction_284.png']
paths=[os.path.join(fig_dir,p) for p in imgs]
for p in paths:
    if not os.path.exists(p):
        raise SystemExit('Missing: '+p)
ims=[Image.open(p).convert('RGBA') for p in paths]
w=600
resized=[im.resize((w,int(im.height* w/im.width))) for im in ims]
w_total=w*2
h_total=resized[0].height+resized[2].height
new=Image.new('RGBA',(w_total,h_total),(255,255,255,255))
new.paste(resized[0],(0,0))
new.paste(resized[1],(w,0))
new.paste(resized[2],(0,resized[0].height))
new.paste(resized[3],(w,resized[1].height))
out=os.path.join(fig_dir,'case_study.png')
new.convert('RGB').save(out)
print('wrote',out)
