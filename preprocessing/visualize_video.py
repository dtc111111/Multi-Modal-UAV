# coding: UTF-8
# author: songyz2019

import os

from moviepy.editor import ImageClip, concatenate_videoclips, TextClip, CompositeVideoClip
from pathlib import Path

IMG_PATH = Path('./seqs/seq0014/Image')
OUT_PATH = Path('video.mp4')
FPS=30

clips = []
for i in sorted(IMG_PATH.iterdir()):
    img_clip = ImageClip(str(i)).set_duration(1/FPS).set_fps(FPS)
    tet_clip = TextClip(txt=str(i), fontsize=50, color='white').set_position(('right','top')).set_duration(img_clip.duration)
    clips.append(CompositeVideoClip([img_clip, tet_clip]))
final_clip = concatenate_videoclips(clips, method="compose")
final_clip.write_videofile(str(OUT_PATH), fps=FPS)