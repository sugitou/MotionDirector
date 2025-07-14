#!/bin/bash

python MotionDirector_inference.py \
  --model ./models/zeroscope_v2_576w/ \
  --prompt "A monkey is playing golf in Times Square New York." \
  --checkpoint_folder ./outputs/myTrain/train_ucf101_golf_8fps_A_person_is_playing_golf_on_the_grass \
  --checkpoint_index 400 \
  --noise_prior 0. \
  --seed 785490 \
  --num-steps 100 \