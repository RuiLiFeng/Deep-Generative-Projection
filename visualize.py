import numpy as np
from PIL import Image
import imageio

def save_img(img_np, save_path):
    Image.fromarray(img_np).save(save_path)
    return 

def save_video(outdir, video_name, ref_img_1, ref_img_2, img_list):

    video = imageio.get_writer(f'{outdir}{video_name}', mode='I', fps=10, codec='libx264', bitrate='16M')
    print (f'Saving optimization progress video "{outdir}{video_name}"')

    for img in img_list:
        video.append_data(np.concatenate([ref_img_1, ref_img_2, img], axis=1))
    video.close()
