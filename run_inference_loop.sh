#!/bin/bash

# Define the checkpoints and their corresponding prompts
declare -A checkpoint_prompts

prompt_text=$(
cat <<EOL
A man wearing white tank top practices cooking, flipping pancakes in his garage home gym.
A knight equipping armor practices boxing, punching a red heavy bag in his garage home gym.
A man wearing white tank top practices boxing, punching a red heavy bag in a boxing ring.
EOL
)
checkpoint_prompts["./outputs/myTrain/train_ucf101_BoxingPunchingBag-short"]="$prompt_text"

prompt_text=$(
cat <<EOL
A man is practising his football dribble on the green front lawn of a brick house.
A wizard is practising his golf swing on the green front lawn of a brick house.
A man is practising his golf swing in Times Square New York.
EOL
)
checkpoint_prompts["./outputs/myTrain/train_ucf101_GolfSwing-short"]="$prompt_text"

prompt_text=$(
cat <<EOL
A skater with a red backpack is handstand walking on a paved road next to a snowy mountain.
An emperor penguin with a red backpack is carving on a paved road next to a snowy mountain.
A skater with a red backpack is carving on a paved road in front of Tokyo Skytree.
EOL
)
checkpoint_prompts["./outputs/myTrain/train_ucf101_SkateBoarding-short"]="$prompt_text"

prompt_text=$(
cat <<EOL
A young girl with long blonde hair is applying lipstick to her lips in a bathroom.
A crystal skeleton is brushing its teeth in a bathroom.
A young girl with long blonde hair is brushing her teeth at Heathrow Airport.
EOL
)
checkpoint_prompts["./outputs/myTrain/train_ucf101_BrushingTeeth-short"]="$prompt_text"

prompt_text=$(
cat <<EOL
A strong man does jumping jacks with perfect form in a gym.
A muscular alien performs an impressive snatch, lifting a heavy barbell overhead in a gym.
A strong man performs an impressive snatch, lifting a heavy barbell overhead in front of Buckingham Palace.
EOL
)
checkpoint_prompts["./outputs/myTrain/train_ucf101_CleanAndJerk-short"]="$prompt_text"

prompt_text=$(
cat <<EOL
Two focused fencers practice footwork on a strip inside a large sports hall.
Two focused women in kimonos duel on a strip inside a large sports hall.
Two focused fencers in white uniforms duel on a strip inside a business office.
EOL
)
checkpoint_prompts["./outputs/myTrain/train_ucf101_Fencing-short"]="$prompt_text"

prompt_text=$(
cat <<EOL
An athletic person does a yoga pose on a grassy hill under a vast blue sky.
A silver robot holds a steady handstand on a grassy hill under a vast blue sky.
An athletic person holds a steady handstand on a sandy beach under a few white cloud sky.
EOL
)
checkpoint_prompts["./outputs/myTrain/train_ucf101_HandstandWalking-short"]="$prompt_text"

prompt_text=$(
cat <<EOL
A rider on horseback jumps over fences in a sandy arena with trees and hills.
A man riding a zebra navigates an obstacle course in a sandy arena with trees and hills.
A rider on horseback navigates an obstacle course on the Moon, with Earth visible in the sky.
EOL
)
checkpoint_prompts["./outputs/myTrain/train_ucf101_HorseRiding-short"]="$prompt_text"

# noise_prior values to iterate over
noise_priors=(0.1 0.3 0.5)

for checkpoint_folder in "${!checkpoint_prompts[@]}"; do
  checkpoint_name=$(basename "$checkpoint_folder")
  short_name=$(echo "$checkpoint_name" | sed -e 's/^train_ucf101_//' -e 's/-short$//')
  readarray -t prompts <<< "${checkpoint_prompts[$checkpoint_folder]}"
  for noise_prior in "${noise_priors[@]}"; do
    output_dir="./outputs/inference/${short_name}/noise${noise_prior}"
    mkdir -p "$output_dir"
    LOG_FILE="${output_dir}/inference_$(date '+%Y%m%d_%H%M%S').log"

    # Run inference for each prompt
    for prompt in "${prompts[@]}"; do
      python MotionDirector_inference.py \
        --model ./models/zeroscope_v2_576w/ \
        --prompt "$prompt" \
        --checkpoint_folder "$checkpoint_folder" \
        --checkpoint_index 400 \
        --noise_prior "$noise_prior" \
        --seed 785490 \
        --num-frames 24 \
        --num-steps 100 \
        --width 256 \
        --height 192 \
        --guidance-scale 12 \
        --output_dir "$output_dir" \

      # Log the experiment details
      echo "==== Experiment ====" >> "$LOG_FILE"
      echo "datetime: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
      echo "checkpoint_folder: $checkpoint_folder" >> "$LOG_FILE"
      echo "prompt: $prompt" >> "$LOG_FILE"
      echo "noise_prior: $noise_prior" >> "$LOG_FILE"
      echo "output_dir: $output_dir" >> "$LOG_FILE"
      echo "==== End ====" >> "$LOG_FILE"
      echo "" >> "$LOG_FILE"
    done
  done
done
