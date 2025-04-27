#!/bin/bash

python3 MotionDirector_inference_multi.py \
  --model ./models/zeroscope_v2_576w/ \
  --prompt "A person is riding a bicycle in the forest." \
  --spatial_path_folder ./outputs/myTrain/train_2025-04-27T15-09-24/checkpoint-500/spatial/lora \
  --temporal_path_folder ./outputs/myTrain/train_2025-04-27T15-09-24/checkpoint-500/temporal/lora \
  --noise_prior 0.5 \
  --seed 5057764