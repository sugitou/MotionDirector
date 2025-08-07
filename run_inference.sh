#!/bin/bash

python MotionDirector_inference.py \
  --model ./models/zeroscope_v2_576w/ \
  --prompt "A small robot carefully applies eyeliner with a thin brush in her bathroom." \
  --checkpoint_folder ./outputs/myTrain/train_ucf101_ApplyEyeMakeup-short \
  --checkpoint_index 400 \
  --noise_prior 0.5 \
  --seed 785490 \
  --num-frames 24 \
  --num-steps 100 \
  --width 256 \
  --height 192 \
  --guidance-scale 12 \