#!/bin/bash

python3 MotionDirector_inference.py \
  --model ./models/zeroscope_v2_576w/ \
  --prompt "A person is riding a bicycle in a garden." \
  --checkpoint_folder ./outputs/myTrain/train_2025-04-27T15-59-59 \
  --checkpoint_index 500 \
  --noise_prior 0. \
  --seed 7192280