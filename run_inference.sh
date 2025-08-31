#!/bin/bash

python MotionDirector_inference.py \
  --model ./models/zeroscope_v2_576w/ \
  --prompt "A crystal skeleton is brushing its teeth in a bathroom." \
  --checkpoint_folder ./outputs/myTrain/train_2025-08-15T17-07-49 \
  --checkpoint_index 400 \
  --noise_prior 0.1 \
  --seed 785490 \
  --num-frames 120 \
  --num-steps 100 \
  --width 256 \
  --height 192 \
  --guidance-scale 12 \