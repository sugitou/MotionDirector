#!/bin/bash

python MotionDirector_inference.py \
  --model ./models/zeroscope_v2_576w/ \
  --prompt "An athletic person holds a steady handstand on a sandy beach under a few white cloud sky." \
  --checkpoint_folder ./outputs/myTrain/train_ucf101_HandstandWalking-short \
  --checkpoint_index 400 \
  --noise_prior 0.3 \
  --seed 785490 \
  --num-frames 24 \
  --num-steps 100 \
  --width 256 \
  --height 192 \
  --guidance-scale 12 \
  --output_dir ./outputs/inference/HandstandWalking/noise0.3 \