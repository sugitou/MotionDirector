#!/bin/bash

python MotionDirector_inference.py \
  --model ./models/zeroscope_v2_576w/ \
  --prompt "A man wearing white tank top practices cooking, flipping pancakes in his garage home gym." \
  --checkpoint_folder ./outputs/myTrain/train_Boxing_160f \
  --checkpoint_index 400 \
  --noise_prior 0.1 \
  --seed 785490 \
  --num-frames 160 \
  --num-steps 100 \
  --width 256 \
  --height 192 \
  --guidance-scale 12 \