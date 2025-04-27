#!/bin/bash

python3 MotionDirector_inference.py \
  --model ./models/zeroscope_v2_576w/ \
  --prompt "A tank is running on the moon." \
  --checkpoint_folder ./outputs/train/car_16 \
  --checkpoint_index 200 \
  --noise_prior 0.5 \
  --seed 8551187