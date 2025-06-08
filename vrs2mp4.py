import os
import numpy as np
import cv2
from moviepy import ImageSequenceClip
from pyvrs import SyncVRSReader, RecordableTypeId

# 入力VRSファイルと出力先ディレクトリ
vrs_file = "../nymeria_dataset/20230607_s0_james_johnson_act0_e72nhq/recording_head/data/data.vrs"
out_dir = "nymeria_frames"
os.makedirs(out_dir, exist_ok=True)

# VRSファイルを開く
reader = SyncVRSReader(vrs_file)

# Check if the VRS file is valid
for sid in reader.stream_ids:
    print(f"stream_id: {sid}")
for sty in reader.record_types:
    print(f"record_type: {sty}")

# フレーム抽出
frame_paths = []
index = 0
for frame in reader.read_stream(stream_id):
    img = frame.image().as_numpy()
    path = os.path.join(out_dir, f"{index:05d}.jpg")
    cv2.imwrite(path, img)
    frame_paths.append(path)
    index += 1


# MP4に変換
clip = ImageSequenceClip(frame_paths, fps=30)
clip.write_videofile("output.mp4", codec="libx264")

