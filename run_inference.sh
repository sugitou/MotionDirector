#!/bin/bash

python3 MotionDirector_inference.py \
  --model ./models/zeroscope_v2_576w/ \
  --prompt "A panda is playing basketball with another panda in a cozy living room." \
  --checkpoint_folder ./outputs/myTrain/train_2025-06-23T14-16-03 \
  --checkpoint_index 150 \
  --noise_prior 0. \
  --seed 7192280