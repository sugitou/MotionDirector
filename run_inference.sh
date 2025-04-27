#!/bin/bash

python3 MotionDirector_inference.py \
  --model ./models/zeroscope_v2_576w/ \
  --prompt "A tank is running on the moon." \
  --checkpoint_folder ./outputs/myTrain/train_2025-04-27T17-02-47 \
  --checkpoint_index 150 \
  --noise_prior 0.5 \
  --seed 8808231