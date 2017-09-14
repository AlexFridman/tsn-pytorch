import os
import subprocess

from tqdm import tqdm_notebook

FNULL = open(os.devnull, 'w')

def extract_rgb_frames(args):
    is_ok = True
    src_video_path, dst_frames_template_path, fps, width, height = args
    try:
        subprocess.call(['ffmpeg', '-r', str(int(fps)), '-i', src_video_path, '-vf',
                         'scale={}:{}'.format(width, height), dst_frames_template_path], stdout=FNULL, stderr=FNULL)
    except Exception as e:
        is_ok = False
        print(e)
    
    return is_ok, src_video_path

def ensure_path_exists(path):
    os.makedirs(path, exist_ok=True)