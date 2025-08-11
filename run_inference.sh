#!/bin/bash

python MotionDirector_inference.py \
  --model ./models/zeroscope_v2_576w/ \
<<<<<<< HEAD
  --prompt "A man wearing white tank top practices cooking, flipping pancakes in his garage home gym." \
  --checkpoint_folder ./outputs/myTrain/train_ucf101_BoxingPunchingBag-short \
=======
  --prompt "Two focused fencers in white uniforms duel on a strip inside a business office." \
  --checkpoint_folder ./outputs/myTrain/train_Fencing_80f \
>>>>>>> 6b9b6ed... Try on HPC
  --checkpoint_index 400 \
  --noise_prior 0.1 \
  --seed 785490 \
  --num-frames 80 \
  --num-steps 100 \
  --width 256 \
  --height 192 \
  --guidance-scale 12 \