#!/bin/bash

python3 MotionDirector_inference.py \
  --model ./models/zeroscope_v2_576w/ \
  --prompt "A car is running on the road." \
  --checkpoint_folder ./outputs/myTrain/train_2025-04-27T12-43-03 \
  --checkpoint_index 150 \
  --noise_prior 0.