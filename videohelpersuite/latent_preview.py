import os
import subprocess

from PIL import Image
import time
import io
import struct
from threading import Thread
import torch.nn.functional as F
import torch

import folder_paths
import numpy as np
import cv2
import shutil

import latent_preview
import server
serv = server.PromptServer.instance

from .utils import hook

rates_table = {'Mochi': 24//6, 'LTXV': 24//8, 'HunyuanVideo': 24//4,
               'Cosmos1CV8x8x8': 24//8, 'Wan21': 16//4, 'Wan22': 24//4}

class WrappedPreviewer(latent_preview.LatentPreviewer):
    def __init__(self, previewer, rate=8):
        self.first_preview = True
        self.last_time = 0
        self.c_index = 0
        self.rate = rate
        if hasattr(previewer, 'taesd'):
            self.taesd = previewer.taesd
        elif hasattr(previewer, 'latent_rgb_factors'):
            self.latent_rgb_factors = previewer.latent_rgb_factors
            self.latent_rgb_factors_bias = previewer.latent_rgb_factors_bias
            self.latent_rgb_factors_reshape = getattr(previewer, 'latent_rgb_factors_reshape', None)
        else:
            raise Exception('Unsupported preview type for VHS animated previews')

    def decode_latent_to_preview_image(self, preview_format, x0, step):
        print(f"ndim: {x0.ndim}, shape: {x0.shape}")
        if x0.ndim == 5:
            num_images = (x0.size(2) - 1) * 4 + x0.size(0) # account for 4x temporal upsampling and batch size

        else:
            num_images = x0.size(0)

        new_time = time.time()
        num_previews = int((new_time - self.last_time) * self.rate)
        self.last_time = self.last_time + num_previews/self.rate
        if num_previews > num_images:
            num_previews = num_images
        elif num_previews <= 0:
            return None
        if self.first_preview:
            self.first_preview = False
            serv.send_sync('VHS_latentpreview', {'length':num_images, 'rate': self.rate, 'id': serv.last_node_id})
            self.last_time = new_time + 1/self.rate

        if self.c_index + num_previews > num_images:
            if x0.ndim == 5:
                x0 = x0.roll(-self.c_index, 1)[:,:num_previews]
            else:
                x0 = x0.roll(-self.c_index, 0)[:num_previews]
        else:
            if x0.ndim == 5:
                x0 = x0[:,self.c_index:self.c_index + num_previews]
            else:
                x0 = x0[self.c_index:self.c_index + num_previews]

        Thread(target=self.process_previews, args=(x0, self.c_index,
                                                   num_images, step)).run()
        self.c_index = (self.c_index + num_previews) % num_images
        return None
    
    def process_previews(self, image_tensor, ind, leng, step):
        image_tensor = self.decode_latent_to_preview(image_tensor)

        if image_tensor.ndim == 5:
            # (B, H, W, T, C) → (B*T, H, W, C)
            B, H, W, T, C = image_tensor.shape
            image_tensor = image_tensor.permute(0, 3, 1, 2, 4)  # (B, T, H, W, C)
            image_tensor = image_tensor.reshape(B*T, H, W, C)

        if image_tensor.size(1) > 512 or image_tensor.size(2) > 512:
            image_tensor = image_tensor.movedim(-1,0)
            if image_tensor.size(2) < image_tensor.size(3):
                height = (512 * image_tensor.size(2)) // image_tensor.size(3)
                image_tensor = F.interpolate(image_tensor, (height,512), mode='bilinear')
            else:
                width = (512 * image_tensor.size(3)) // image_tensor.size(2)
                image_tensor = F.interpolate(image_tensor, (512, width), mode='bilinear')
            image_tensor = image_tensor.movedim(0,-1)
        previews_ubyte = (((image_tensor + 1.0) / 2.0).clamp(0, 1)  # change scale from -1..1 to 0..1
                         .mul(0xFF)  # to 0..255
                         ).to(device="cpu", dtype=torch.uint8)

        img_id = 0
        for preview in previews_ubyte:
            i = Image.fromarray(preview.numpy())
            message = io.BytesIO()
            message.write((1).to_bytes(length=4, byteorder='big')*2)
            message.write(ind.to_bytes(length=4, byteorder='big'))
            message.write(struct.pack('16p', serv.last_node_id.encode('ascii')))
            i.save(message, format="JPEG", quality=95, compress_level=1)
            #NOTE: send sync already uses call_soon_threadsafe

            serv.send_sync(server.BinaryEventTypes.PREVIEW_IMAGE,
                           message.getvalue(), serv.client_id)
            ind = (ind + 1) % leng

        self.save_preview_video(previews_ubyte, step)

        
    def save_preview_video(self, previews_ubyte, step):
        # Save as video via ffmpeg, overwriting each step
        video_dir = os.path.join(folder_paths.get_output_directory(), "previews")
        os.makedirs(video_dir, exist_ok=True)
        current_video_files = [f for f in os.listdir(video_dir) if f.startswith(f"preview") and f.endswith(".mp4")]
        
        video_name = "preview.mp4"
        
        if len(current_video_files) > 0:
            # sort by modification time, newest first
            current_video_files.sort(key=lambda f: os.path.getmtime(os.path.join(video_dir, f)), reverse=True)
            _, prev_counter, prev_step = os.path.splitext(current_video_files[0])[0].split("_")

            if step == int(prev_step) + 1:
                # remove the existing video to overwrite it
                os.remove(os.path.join(video_dir, current_video_files[0]))
                video_name = f"preview_{prev_counter}_{step}.mp4"
            else:
                video_name = f"preview_{int(prev_counter)+1}_{step}.mp4"

        else:
            video_name = f"preview_1_{step}.mp4"

        video_path = os.path.join(video_dir, video_name)

        h, w = previews_ubyte.shape[1], previews_ubyte.shape[2]
        w = w if w % 2 == 0 else w - 1
        h = h if h % 2 == 0 else h - 1
        previews_ubyte = previews_ubyte[:, :h, :w, :]

        ffmpeg_command = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{w}x{h}",
            "-pix_fmt", "rgb24",
            "-r", "15",
            "-i", "pipe:0",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "28",
            "-pix_fmt", "yuv420p",
            video_path
        ]

        proc = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for frame in previews_ubyte:
            proc.stdin.write(frame.numpy().tobytes())
        proc.stdin.close()
        proc.wait()

    def decode_latent_to_preview(self, x0):
        if hasattr(self, 'taesd'):
            x_sample = self.taesd.decode(x0).movedim(1, 3)
            return x_sample
        else:
            if self.latent_rgb_factors_reshape is not None:
                x0 = self.latent_rgb_factors_reshape(x0)
            self.latent_rgb_factors = self.latent_rgb_factors.to(dtype=x0.dtype, device=x0.device)
            if self.latent_rgb_factors_bias is not None:
                self.latent_rgb_factors_bias = self.latent_rgb_factors_bias.to(dtype=x0.dtype, device=x0.device)
            latent_image = F.linear(x0.movedim(1, -1), self.latent_rgb_factors,
                                    bias=self.latent_rgb_factors_bias)
            return latent_image

@hook(latent_preview, 'get_previewer')
def get_latent_video_previewer(device, latent_format, *args, **kwargs):
    node_id = serv.last_node_id
    previewer = get_latent_video_previewer.__wrapped__(device, latent_format, *args, **kwargs)
    try:
        extra_info = next(serv.prompt_queue.currently_running.values().__iter__()) \
                [3]['extra_pnginfo']['workflow']['extra']
        prev_setting = extra_info.get('VHS_latentpreview', False)
        if extra_info.get('VHS_latentpreviewrate', 0) != 0:
            rate_setting = extra_info['VHS_latentpreviewrate']
        else:
            rate_setting = rates_table.get(latent_format.__class__.__name__, 8)
    except:
        #For safety since there's lots of keys, any of which can fail
        prev_setting = False
    if not prev_setting or not hasattr(previewer, "decode_latent_to_preview"):
        return previewer
    return WrappedPreviewer(previewer, rate_setting)
