#!/bin/bash

# Define the checkpoints and their corresponding prompts
declare -A checkpoint_prompts

prompt_text=$(
cat <<EOL
A man wearing white tank top practices cooking, flipping pancakes in his garage home gym.
A man wearing white tank top practices boxing, punching a red heavy bag in a boxing ring.
EOL
)
checkpoint_prompts["./outputs/myTrain/train_Boxing_120f_linear"]="$prompt_text"

prompt_text=$(
cat <<EOL
A man is practising his football dribble on the green front lawn of a brick house.
A man is practising his golf swing in Times Square New York.
EOL
)
checkpoint_prompts["./outputs/myTrain/train_Golf_120f_linear"]="$prompt_text"

prompt_text=$(
cat <<EOL
A focused woman carefully applies face cream with gentle motions in her bathroom.
A small robot carefully applies eyeliner with a thin brush in her bathroom.
EOL
)
checkpoint_prompts["./outputs/myTrain/train_Makeup_120f_linear"]="$prompt_text"

prompt_text=$(
cat <<EOL
A young girl with long blonde hair is applying lipstick to her lips in a bathroom.
A crystal skeleton is brushing its teeth in a bathroom.
EOL
)
checkpoint_prompts["./outputs/myTrain/train_Brushing_120f_linear"]="$prompt_text"

prompt_text=$(
cat <<EOL
A muscular alien performs an impressive snatch, lifting a heavy barbell overhead in a gym.
A strong man performs an impressive snatch, lifting a heavy barbell overhead in front of Buckingham Palace.
EOL
)
checkpoint_prompts["./outputs/myTrain/train_CleanAndJerk_120f_linear"]="$prompt_text"

prompt_text=$(
cat <<EOL
Two focused fencers in white uniforms duel on a strip inside a business office.
EOL
)
checkpoint_prompts["./outputs/myTrain/train_Fencing_120f_linear"]="$prompt_text"

prompt_text=$(
cat <<EOL
A silver robot holds a steady handstand on a grassy hill under a vast blue sky.
An athletic person holds a steady handstand on a sandy beach under a few white cloud sky.
EOL
)
checkpoint_prompts["./outputs/myTrain/train_Hand_120f_linear"]="$prompt_text"

prompt_text=$(
cat <<EOL
A climber ascends a challenging ice wall with a follow cinematic shot.
A woman in a maid's costume ascends a challenging indoor climbing wall with a follow cinematic shot.
EOL
)
checkpoint_prompts["./outputs/myTrain/train_Climbing_120f_linear"]="$prompt_text"

### 2) 長時間動画の共通設定（必要に応じて調整）
MODEL="./models/zeroscope_v2_576w/"
CKPT_INDEX=400
SEED=785490
NUM_FRAMES=80           # ← 長時間化（例：96フレーム）
NUM_STEPS=100           # ← より丁寧にサンプル
WIDTH=256
HEIGHT=192
GUIDANCE=12
TARGET_BASE="./outputs/inference_120f"   # 長時間用の保存ルート

### 3) 「明示したペア」だけを回す： checkpointごとのペア列挙
# 形式: "promptIndex:noise promptIndex:noise ..."
declare -A prompt_noise_pairs
# prompt_noise_pairs["./outputs/myTrain/train_Boxing_120f_linear"]="0:0.1 1:0.5"
# prompt_noise_pairs["./outputs/myTrain/train_Golf_120f_linear"]="0:0.5 1:0.5"
# prompt_noise_pairs["./outputs/myTrain/train_Makeup_120f_linear"]="0:0.3 1:0.5"
# prompt_noise_pairs["./outputs/myTrain/train_Brushing_120f_linear"]="0:0.1 1:0.3 1:0.1"
# prompt_noise_pairs["./outputs/myTrain/train_CleanAndJerk_120f_linear"]="0:0.3 1:0.5"
# prompt_noise_pairs["./outputs/myTrain/train_Fencing_120f_linear"]="0:0.1 0:0.5"
prompt_noise_pairs["./outputs/myTrain/train_Hand_120f_linear"]="0:0.1 1:0.3"
# prompt_noise_pairs["./outputs/myTrain/train_Climbing_120f_linear"]="0:0.1 1:0.1"

### 4) 実行順を固定したい場合は配列で列挙
checkpoints=(
  "./outputs/myTrain/train_Makeup_120f_linear"
  "./outputs/myTrain/train_Boxing_120f_linear"
  "./outputs/myTrain/train_Brushing_120f_linear"
  "./outputs/myTrain/train_CleanAndJerk_120f_linear"
  "./outputs/myTrain/train_Fencing_120f_linear"
  "./outputs/myTrain/train_Golf_120f_linear"
  "./outputs/myTrain/train_Hand_120f_linear"
  "./outputs/myTrain/train_Climbing_120f_linear"
)

### 5) ループ本体（明示ペアのみ）
for checkpoint_folder in "${checkpoints[@]}"; do
  pairs="${prompt_noise_pairs[$checkpoint_folder]:-}"
  [[ -z "$pairs" ]] && { echo "[SKIP] No pairs for $checkpoint_folder"; continue; }

  checkpoint_name=$(basename "$checkpoint_folder")  # train_ucf101_XXX-short
  short_name=$(echo "$checkpoint_name" | sed -e 's/^train_//' -e 's/_80f_linear$//')

  # そのcheckpointの3つのpromptを配列化
  readarray -t prompts <<< "${checkpoint_prompts[$checkpoint_folder]}"

  # ペア列挙を回す（例: "1:0.5"）
  for pair in $pairs; do
    IFS=':' read -r pidx noise_prior <<< "$pair"

    # インデックス範囲チェック
    if (( pidx < 0 || pidx >= ${#prompts[@]} )); then
      echo "[WARN] Invalid prompt_index=$pidx for $checkpoint_folder (has ${#prompts[@]} prompts). Skip."
      continue
    fi

    prompt="${prompts[$pidx]}"
    out_dir="${TARGET_BASE}/${short_name}/noise${noise_prior}"
    mkdir -p "$out_dir"

    # ログ/対応表
    LOG_FILE="${out_dir}/inference_$(date '+%Y%m%d_%H%M%S')_p${pidx}.log"
    PARAMS_FILE="${out_dir}/params.txt"
    echo "p${pidx}: ${prompt}" >> "$PARAMS_FILE"

    # 推論コマンド（ファイル名はスクリプト側仕様に依存して自動）
    python MotionDirector_inference.py \
      --model "$MODEL" \
      --prompt "$prompt" \
      --checkpoint_folder "$checkpoint_folder" \
      --checkpoint_index "$CKPT_INDEX" \
      --noise_prior "$noise_prior" \
      --seed "$SEED" \
      --num-frames "$NUM_FRAMES" \
      --num-steps "$NUM_STEPS" \
      --width "$WIDTH" \
      --height "$HEIGHT" \
      --guidance-scale "$GUIDANCE" \
      --output_dir "$out_dir" \
      >> "$LOG_FILE" 2>&1

    {
      echo "==== Experiment ===="
      echo "datetime: $(date '+%Y-%m-%d %H:%M:%S')"
      echo "checkpoint_folder: $checkpoint_folder"
      echo "prompt_index: $pidx"
      echo "prompt: $prompt"
      echo "noise_prior: $noise_prior"
      echo "num_frames: $NUM_FRAMES"
      echo "num_steps: $NUM_STEPS"
      echo "output_dir: $out_dir"
      echo "==== End ===="
      echo
    } >> "$LOG_FILE"
  done
done