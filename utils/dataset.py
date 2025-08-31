import os
import decord
import numpy as np
import random
import json
import torchvision
import torchvision.transforms as T
import torch

from glob import glob
from PIL import Image
from itertools import islice
from pathlib import Path
from .bucketing import sensible_buckets

decord.bridge.set_bridge('torch')

from torch.utils.data import Dataset
from einops import rearrange, repeat

import cv2


# Interpolation for temporal resampling
def time_resample_linear(video: torch.Tensor, target_frames: int) -> torch.Tensor:
    """
    video: [N, C, H, W] (uint8/float いずれも可)
    target_frames: 出力フレーム数T
    return: [T, C, H, W] (float32)
    """
    assert video.ndim == 4, "video must be [N, C, H, W]"
    N, C, H, W = video.shape
    T = int(target_frames)
    assert T > 0, "target_frames must be > 0"

    if N == T:
        return video if video.dtype == torch.float32 else video.to(torch.float32)
    if N == 1:
        return video.to(torch.float32).repeat(T, 1, 1, 1)

    dev = video.device
    vid = video.to(torch.float32, copy=False)

    u = torch.linspace(0.0, float(N - 1), steps=T, device=dev)  # [0, N-1]を等分
    i = torch.floor(u).to(torch.long)
    j = torch.clamp(i + 1, max=N - 1)
    a = (u - i.to(u.dtype)).view(T, 1, 1, 1)  # [T,1,1,1]

    out = (1.0 - a) * vid[i] + a * vid[j]
    return out


# Flow-based temporal resampling
def time_resample_flow(video: torch.Tensor, target_frames: int,
                       algo: str = "farneback",
                       symmetric_blend: bool = True) -> torch.Tensor:
    """
    video: [N, C, H, W] (uint8/float) 0..255 想定
    target_frames: 出力フレーム数 T
    algo: "farneback"（軽量・依存なし）
    symmetric_blend:
        True  -> Fiを前方に a 倍ワープ、Fjを後方に(1-a)倍ワープして時間重みでブレンド
        False -> Fi を前方に a 倍だけワープして線形ブレンド（片側のみ; 速い）

    戻り値: [T, C, H, W] (float32, 0..255)
    """
    assert video.ndim == 4
    N, C, H, W = video.shape
    T = int(target_frames)
    assert T > 0

    # 早期リターン
    if N == T:
        return video.to(torch.float32)
    if N == 1:
        return video.to(torch.float32).repeat(T, 1, 1, 1)

    # Tensor -> numpy (HxWxC, BGRじゃなくてもOK。cv2.remapは色順気にしない)
    vid_np = video.detach().cpu().to(torch.uint8).permute(0, 2, 3, 1).numpy()  # [N, H, W, C]
    # 光フローはグレイでOK
    gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) if vid_np.shape[3] == 3 else f[...,0] for f in vid_np]

    # 事前にペアごとのフローを計算（i->i+1 と (i+1)->i）
    flows_fwd = []  # [N-1, H, W, 2]
    flows_bwd = []  # [N-1, H, W, 2]
    # Farneback推奨設定（256x192程度なら十分高速）
    fb_params = dict(pyr_scale=0.5, levels=3, winsize=15,
                     iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

    for i in range(N - 1):
        g0, g1 = gray[i], gray[i + 1]
        flow_f = cv2.calcOpticalFlowFarneback(g0, g1, None, **fb_params)  # g0 -> g1
        flow_b = cv2.calcOpticalFlowFarneback(g1, g0, None, **fb_params)  # g1 -> g0
        flows_fwd.append(flow_f)
        flows_bwd.append(flow_b)

    flows_fwd = np.stack(flows_fwd, axis=0)  # [N-1, H, W, 2]
    flows_bwd = np.stack(flows_bwd, axis=0)  # [N-1, H, W, 2]

    # 座標グリッド（remap用）
    grid_x, grid_y = np.meshgrid(np.arange(W, dtype=np.float32),
                                 np.arange(H, dtype=np.float32))

    # 出力を合成
    out = np.empty((T, H, W, C), dtype=np.float32)
    for t in range(T):
        u = (N - 1) * t / max(1, T - 1)      # 連続時刻 u ∈ [0, N-1]
        i = int(np.floor(u))
        if i >= N - 1:
            out[t] = vid_np[-1].astype(np.float32)
            continue
        a = float(u - i)                     # 0..1
        j = i + 1

        # 参照フレーム
        Fi = vid_np[i].astype(np.float32)    # [H,W,C]
        Fj = vid_np[j].astype(np.float32)

        # フロー取得
        fwd = flows_fwd[i]                   # [H,W,2] (x,y の順)
        if symmetric_blend:
            bwd = flows_bwd[i]
        else:
            bwd = None

        # 前方/後方のワープ量をスケール
        map_x_i = grid_x + a * fwd[..., 0]
        map_y_i = grid_y + a * fwd[..., 1]

        if symmetric_blend:
            map_x_j = grid_x + (1.0 - a) * bwd[..., 0]
            map_y_j = grid_y + (1.0 - a) * bwd[..., 1]

        # remap（境界は複製でOK）
        warp_i = cv2.remap(Fi, map_x_i, map_y_i, interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
        if symmetric_blend:
            warp_j = cv2.remap(Fj, map_x_j, map_y_j, interpolation=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
            # 時間重みでブレンド（aに応じてFi→Fjへ遷移）
            out[t] = (1.0 - a) * warp_i + a * warp_j
        else:
            # 片側のみ：Fiを未来側へ動かし、Fjと線形ブレンド
            out[t] = (1.0 - a) * warp_i + a * Fj

    # numpy -> torch [T,C,H,W]
    return torch.from_numpy(out).permute(0, 3, 1, 2)  # float32


def get_prompt_ids(prompt, tokenizer):
    prompt_ids = tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
    ).input_ids

    return prompt_ids


def read_caption_file(caption_file):
        with open(caption_file, 'r', encoding="utf8") as t:
            return t.read()


def get_text_prompt(
        text_prompt: str = '', 
        fallback_prompt: str= '',
        file_path:str = '', 
        ext_types=['.mp4'],
        use_caption=False
    ):
    try:
        if use_caption:
            if len(text_prompt) > 1: return text_prompt
            caption_file = ''
            # Use caption on per-video basis (One caption PER video)
            for ext in ext_types:
                maybe_file = file_path.replace(ext, '.txt')
                if maybe_file.endswith(ext_types): continue
                if os.path.exists(maybe_file): 
                    caption_file = maybe_file
                    break

            if os.path.exists(caption_file):
                return read_caption_file(caption_file)
            
            # Return fallback prompt if no conditions are met.
            return fallback_prompt

        return text_prompt
    except:
        print(f"Couldn't read prompt caption for {file_path}. Using fallback.")
        return fallback_prompt

    
def get_video_frames(vr, start_idx, sample_rate=1, max_frames=24):
    max_range = len(vr)
    frame_number = sorted((0, start_idx, max_range))[1]

    frame_range = range(frame_number, max_range, sample_rate)
    frame_range_indices = list(frame_range)[:max_frames]

    return frame_range_indices


def process_video(vid_path, use_bucketing, w, h, get_frame_buckets, get_frame_batch):
    if use_bucketing:
        vr = decord.VideoReader(vid_path)
        resize = get_frame_buckets(vr)
        video = get_frame_batch(vr, resize=resize)

    else:
        vr = decord.VideoReader(vid_path, width=w, height=h)
        video = get_frame_batch(vr)

    return video, vr


# https://github.com/ExponentialML/Video-BLIP2-Preprocessor
class VideoJsonDataset(Dataset):
    def __init__(
            self,
            tokenizer = None,
            width: int = 256,
            height: int = 256,
            n_sample_frames: int = 4,
            sample_start_idx: int = 1,
            frame_step: int = 1,
            json_path: str ="",
            json_data = None,
            vid_data_key: str = "video_path",
            preprocessed: bool = False,
            use_bucketing: bool = False,
            **kwargs
    ):
        self.vid_types = (".mp4", ".avi", ".mov", ".webm", ".flv", ".mjpeg")
        self.use_bucketing = use_bucketing
        self.tokenizer = tokenizer
        self.preprocessed = preprocessed
        
        self.vid_data_key = vid_data_key
        self.train_data = self.load_from_json(json_path, json_data)

        self.width = width
        self.height = height

        self.n_sample_frames = n_sample_frames
        self.sample_start_idx = sample_start_idx
        self.frame_step = frame_step

    def build_json(self, json_data):
        extended_data = []
        for data in json_data['data']:
            for nested_data in data['data']:
                self.build_json_dict(
                    data, 
                    nested_data, 
                    extended_data
                )
        json_data = extended_data
        return json_data

    def build_json_dict(self, data, nested_data, extended_data):
        clip_path = nested_data['clip_path'] if 'clip_path' in nested_data else None
        
        extended_data.append({
            self.vid_data_key: data[self.vid_data_key],
            'frame_index': nested_data['frame_index'],
            'prompt': nested_data['prompt'],
            'clip_path': clip_path
        })
        
    def load_from_json(self, path, json_data):
        try:
            with open(path) as jpath:
                print(f"Loading JSON from {path}")
                json_data = json.load(jpath)

                return self.build_json(json_data)

        except:
            self.train_data = []
            print("Non-existant JSON path. Skipping.")
            
    def validate_json(self, base_path, path):
        return os.path.exists(f"{base_path}/{path}")

    def get_frame_range(self, vr):
        return get_video_frames(
            vr, 
            self.sample_start_idx, 
            self.frame_step, 
            self.n_sample_frames
        )
    
    def get_vid_idx(self, vr, vid_data=None):
        frames = self.n_sample_frames

        if vid_data is not None:
            idx = vid_data['frame_index']
        else:
            idx = self.sample_start_idx

        return idx

    def get_frame_buckets(self, vr):
        _, h, w = vr[0].shape        
        width, height = sensible_buckets(self.width, self.height, h, w)
        # width, height = self.width, self.height
        resize = T.transforms.Resize((height, width), antialias=True)

        return resize

    def get_frame_batch(self, vr, resize=None):
        frame_range = self.get_frame_range(vr)
        frames = vr.get_batch(frame_range)
        video = rearrange(frames, "f h w c -> f c h w")

        if resize is not None: video = resize(video)
        # Add time resampling
        if video.shape[0] != self.n_sample_frames:
            video = time_resample_linear(video, self.n_sample_frames)
            #video = time_resample_flow(video, self.n_sample_frames, symmetric_blend=True)


        return video

    def process_video_wrapper(self, vid_path):
        video, vr = process_video(
                vid_path,
                self.use_bucketing,
                self.width, 
                self.height, 
                self.get_frame_buckets, 
                self.get_frame_batch
            )
        
        return video, vr 

    def train_data_batch(self, index):

        # If we are training on individual clips.
        if 'clip_path' in self.train_data[index] and \
            self.train_data[index]['clip_path'] is not None:

            vid_data = self.train_data[index]

            clip_path = vid_data['clip_path']
            
            # Get video prompt
            prompt = vid_data['prompt']

            video, _ = self.process_video_wrapper(clip_path)

            prompt_ids = get_prompt_ids(prompt, self.tokenizer)

            return video, prompt, prompt_ids

         # Assign train data
        train_data = self.train_data[index]
        
        # Get the frame of the current index.
        self.sample_start_idx = train_data['frame_index']
        
        # Initialize resize
        resize = None

        video, vr = self.process_video_wrapper(train_data[self.vid_data_key])

        # Get video prompt
        prompt = train_data['prompt']
        vr.seek(0)

        prompt_ids = get_prompt_ids(prompt, self.tokenizer)

        return video, prompt, prompt_ids

    @staticmethod
    def __getname__(): return 'json'

    def __len__(self):
        if self.train_data is not None:
            return len(self.train_data)
        else: 
            return 0

    def __getitem__(self, index):
        
        # Initialize variables
        video = None
        prompt = None
        prompt_ids = None

        # Use default JSON training
        if self.train_data is not None:
            video, prompt, prompt_ids = self.train_data_batch(index)

        example = {
            "pixel_values": (video / 127.5 - 1.0),
            "prompt_ids": prompt_ids[0],
            "text_prompt": prompt,
            'dataset': self.__getname__()
        }

        return example


class SingleVideoDataset(Dataset):
    def __init__(
        self,
            tokenizer = None,
            width: int = 256,
            height: int = 256,
            n_sample_frames: int = 4,
            frame_step: int = 1,
            single_video_path: str = "",
            single_video_prompt: str = "",
            use_caption: bool = False,
            use_bucketing: bool = False,
            **kwargs
    ):
        self.tokenizer = tokenizer
        self.use_bucketing = use_bucketing
        self.frames = []
        self.index = 1

        self.vid_types = (".mp4", ".avi", ".mov", ".webm", ".flv", ".mjpeg")
        self.n_sample_frames = n_sample_frames
        self.frame_step = frame_step

        self.single_video_path = single_video_path
        self.single_video_prompt = single_video_prompt

        self.width = width
        self.height = height
    
    def create_video_chunks(self):
        vr = decord.VideoReader(self.single_video_path)
        vr_range = list(range(0, len(vr), self.frame_step))  # 有効インデックス列

        self.frames = []
        size = self.n_sample_frames
        # 端数チャンクもそのまま保持（ここではpadしない）
        for i in range(0, len(vr_range), size):
            part = vr_range[i:i+size]          # 最後は size 未満になり得る
            if len(part) > 0:
                self.frames.append(tuple(part))
        return self.frames
    
    def get_frame_batch(self, vr, resize=None):
        # 取り出す実フレームindex列
        idxs = list(self.frames[self.index])
        idxs = np.clip(np.array(idxs, dtype=np.int64), 0, len(vr) - 1)

        frames = vr.get_batch(idxs)                          # [f, H, W, C] (uint8)
        video  = rearrange(frames, "f h w c -> f c h w")     # [f, C, H, W]

        # 画質面の安定のため：空間リサイズ → 時間補間 の順がおすすめ
        if resize is not None:
            video = resize(video)

        # ★ ここで常に n_sample_frames に時間リサンプリング（interp）
        if video.shape[0] != self.n_sample_frames:
            video = time_resample_linear(video, self.n_sample_frames)
            #video = time_resample_flow(video, self.n_sample_frames, symmetric_blend=True)

        else:
            if video.dtype != torch.float32:
                video = video.to(torch.float32)

        return video
    
    # def chunk_full(self, it, size, keep_tail=True, pad_mode="pad_last"):
    #     """
    #     it: 反復可能（フレームindexのrangeなど）
    #     size: 1チャンクのフレーム数（= n_sample_frames）
    #     keep_tail: 端数チャンクを保持するか
    #     pad_mode:
    #     - "pad_last":  末尾フレームindexを複製して size に揃える（堅い）
    #     - "loop":      先頭からループして補う（軽い繰り返しを許容）
    #     - None:        端数はそのまま（サイズ未満のタプルのまま返す）
    #     """
    #     it = list(it)
    #     chunks = []
    #     n = len(it)

    #     if n == 0:
    #         return []

    #     for i in range(0, n, size):
    #         part = it[i:i+size]
    #         if len(part) == size:
    #             chunks.append(tuple(part))
    #         else:
    #             if not keep_tail:
    #                 continue
    #             if pad_mode == "pad_last":
    #                 pad = [part[-1]] * (size - len(part))
    #                 part = part + pad
    #             elif pad_mode == "loop":
    #                 need = size - len(part)
    #                 loop_take = min(need, len(it))
    #                 part = part + it[:loop_take]
    #                 if len(part) < size:
    #                     part = part + [part[-1]] * (size - len(part))
    #             elif pad_mode is None:
    #                 # サイズ未満の端数をそのまま保持
    #                 pass
    #             else:
    #                 raise ValueError(f"Unknown pad_mode: {pad_mode}")
    #             chunks.append(tuple(part))
    #     return chunks
    
    # def create_video_chunks(self):
    #     vr = decord.VideoReader(self.single_video_path)
    #     vr_range = range(0, len(vr), self.frame_step)

    #     # 端数も保持し、既定では pad して n_sample_frames に揃える
    #     self.frames = self.chunk_full(
    #         vr_range,
    #         self.n_sample_frames,
    #         keep_tail=True,
    #         pad_mode="pad_last",   # 必要に応じて "loop" や None に変更可
    #     )
    #     return self.frames
    
    # def get_frame_batch(self, vr, resize=None):
    #     # self.frames[self.index] は tuple（または短いtuple）を想定
    #     idxs = np.array(self.frames[self.index], dtype=np.int64)
    #     idxs = np.clip(idxs, 0, len(vr) - 1)
    #     frames = vr.get_batch(idxs)
    #     video = rearrange(frames, "f h w c -> f c h w")
    #     if resize is not None:
    #         video = resize(video)
    #     return video
    
    # Original
    # def create_video_chunks(self):
    #     vr = decord.VideoReader(self.single_video_path)
    #     vr_range = range(0, len(vr), self.frame_step)

    #     self.frames = list(self.chunk(vr_range, self.n_sample_frames))
    #     return self.frames

    # def chunk(self, it, size):
    #     it = iter(it)
    #     return iter(lambda: tuple(islice(it, size)), ())

    # def get_frame_batch(self, vr, resize=None):
    #     index = self.index
    #     frames = vr.get_batch(self.frames[self.index])
    #     video = rearrange(frames, "f h w c -> f c h w")

    #     if resize is not None: video = resize(video)
    #     return video

    def get_frame_buckets(self, vr):
        h, w, c = vr[0].shape
        width, height = sensible_buckets(self.width, self.height, w, h)
        resize = T.transforms.Resize((height, width), antialias=True)

        return resize
    
    def process_video_wrapper(self, vid_path):
        video, vr = process_video(
                vid_path,
                self.use_bucketing,
                self.width, 
                self.height, 
                self.get_frame_buckets, 
                self.get_frame_batch
            )
        
        return video, vr 

    def single_video_batch(self, index):
        train_data = self.single_video_path
        self.index = index

        if train_data.endswith(self.vid_types):
            video, _ = self.process_video_wrapper(train_data)

            prompt = self.single_video_prompt
            prompt_ids = get_prompt_ids(prompt, self.tokenizer)

            return video, prompt, prompt_ids
        else:
            raise ValueError(f"Single video is not a video type. Types: {self.vid_types}")
    
    @staticmethod
    def __getname__(): return 'single_video'

    def __len__(self):
        
        return len(self.create_video_chunks())

    def __getitem__(self, index):

        video, prompt, prompt_ids = self.single_video_batch(index)

        example = {
            "pixel_values": (video / 127.5 - 1.0),
            "prompt_ids": prompt_ids[0],
            "text_prompt": prompt,
            'dataset': self.__getname__()
        }

        return example


class ImageDataset(Dataset):
    
    def __init__(
        self,
        tokenizer = None,
        width: int = 256,
        height: int = 256,
        base_width: int = 256,
        base_height: int = 256,
        use_caption:     bool = False,
        image_dir: str = '',
        single_img_prompt: str = '',
        use_bucketing: bool = False,
        fallback_prompt: str = '',
        **kwargs
    ):
        self.tokenizer = tokenizer
        self.img_types = (".png", ".jpg", ".jpeg", '.bmp')
        self.use_bucketing = use_bucketing

        self.image_dir = self.get_images_list(image_dir)
        self.fallback_prompt = fallback_prompt

        self.use_caption = use_caption
        self.single_img_prompt = single_img_prompt

        self.width = width
        self.height = height

    def get_images_list(self, image_dir):
        if os.path.exists(image_dir):
            imgs = [x for x in os.listdir(image_dir) if x.endswith(self.img_types)]
            full_img_dir = []

            for img in imgs: 
                full_img_dir.append(f"{image_dir}/{img}")

            return sorted(full_img_dir)

        return ['']

    def image_batch(self, index):
        train_data = self.image_dir[index]
        img = train_data

        try:
            img = torchvision.io.read_image(img, mode=torchvision.io.ImageReadMode.RGB)
        except:
            img = T.transforms.PILToTensor()(Image.open(img).convert("RGB"))

        width = self.width
        height = self.height

        if self.use_bucketing:
            _, h, w = img.shape
            width, height = sensible_buckets(width, height, w, h)
              
        resize = T.transforms.Resize((height, width), antialias=True)

        img = resize(img) 
        img = repeat(img, 'c h w -> f c h w', f=16)

        prompt = get_text_prompt(
            file_path=train_data,
            text_prompt=self.single_img_prompt,
            fallback_prompt=self.fallback_prompt,
            ext_types=self.img_types,  
            use_caption=True
        )
        prompt_ids = get_prompt_ids(prompt, self.tokenizer)

        return img, prompt, prompt_ids

    @staticmethod
    def __getname__(): return 'image'
    
    def __len__(self):
        # Image directory
        if os.path.exists(self.image_dir[0]):
            return len(self.image_dir)
        else:
            return 0

    def __getitem__(self, index):
        img, prompt, prompt_ids = self.image_batch(index)
        example = {
            "pixel_values": (img / 127.5 - 1.0),
            "prompt_ids": prompt_ids[0],
            "text_prompt": prompt, 
            'dataset': self.__getname__()
        }

        return example


class VideoFolderDataset(Dataset):
    def __init__(
        self,
        tokenizer=None,
        width: int = 256,
        height: int = 256,
        n_sample_frames: int = 16,
        fps: int = 8,
        path: str = "./data",
        fallback_prompt: str = "",
        use_bucketing: bool = False,
        **kwargs
    ):
        self.tokenizer = tokenizer
        self.use_bucketing = use_bucketing

        self.fallback_prompt = fallback_prompt

        self.video_files = glob(f"{path}/*.mp4")

        self.width = width
        self.height = height

        self.n_sample_frames = n_sample_frames
        self.fps = fps

    def get_frame_buckets(self, vr):
        h, w, c = vr[0].shape
        width, height = sensible_buckets(self.width, self.height, w, h)
        resize = T.transforms.Resize((height, width), antialias=True)

        return resize

    def get_frame_batch(self, vr, resize=None):
        n_sample_frames = self.n_sample_frames
        native_fps = vr.get_avg_fps()
        
        every_nth_frame = max(1, round(native_fps / self.fps))
        every_nth_frame = min(len(vr), every_nth_frame)
        
        effective_length = len(vr) // every_nth_frame
        if effective_length < n_sample_frames:
            n_sample_frames = effective_length

        effective_idx = random.randint(0, (effective_length - n_sample_frames))
        idxs = every_nth_frame * np.arange(effective_idx, effective_idx + n_sample_frames)

        video = vr.get_batch(idxs)
        video = rearrange(video, "f h w c -> f c h w")

        if resize is not None: video = resize(video)
        # Add time resampling
        if video.shape[0] != self.n_sample_frames:
            video = time_resample_linear(video, self.n_sample_frames)
            #video = time_resample_flow(video, self.n_sample_frames, symmetric_blend=True)


        return video, vr
        
    def process_video_wrapper(self, vid_path):
        video, vr = process_video(
                vid_path,
                self.use_bucketing,
                self.width, 
                self.height, 
                self.get_frame_buckets, 
                self.get_frame_batch
            )
        return video, vr
    
    def get_prompt_ids(self, prompt):
        return self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

    @staticmethod
    def __getname__(): return 'folder'

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, index):

        video, _ = self.process_video_wrapper(self.video_files[index])

        prompt = self.fallback_prompt

        prompt_ids = self.get_prompt_ids(prompt)

        return {"pixel_values": (video[0] / 127.5 - 1.0), "prompt_ids": prompt_ids[0], "text_prompt": prompt, 'dataset': self.__getname__()}


class CachedDataset(Dataset):
    def __init__(self,cache_dir: str = ''):
        self.cache_dir = cache_dir
        self.cached_data_list = self.get_files_list()

    def get_files_list(self):
        tensors_list = [f"{self.cache_dir}/{x}" for x in os.listdir(self.cache_dir) if x.endswith('.pt')]
        return sorted(tensors_list)

    def __len__(self):
        return len(self.cached_data_list)

    def __getitem__(self, index):
        cached_latent = torch.load(self.cached_data_list[index], map_location='cuda:0')
        return cached_latent
