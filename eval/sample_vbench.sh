# !/bin/bash

CKPT=$1
NUM_FRAMES=$2
MODEL_NAME=$3
TASK_TYPE=$4
VBENCH_START_INDEX=$5
VBENCH_END_INDEX=$6
VBENCH_RES=$7
VBENCH_ASP_RATIO=$8

NUM_SAMPLING_STEPS=$9
FLOW=${10}
LLM_REFINE=${11}

BASE_ASPECT_RATIO=360p
ASPECT_RATIOS=(144p 240p 360p 480p 720p 1080p)
# Loop through the list of aspect ratios
i=0
for r in "${ASPECT_RATIOS[@]}"; do
  if [[ "$r" == "$BASE_ASPECT_RATIO" ]]; then
    # get aspect ratio 1 level up
    if [[ $((i+1)) -lt ${#ASPECT_RATIOS[@]} ]]; then
      ASPECT_RATIO_INCR_1=${ASPECT_RATIOS[$((i+1))]}
    else
      # If this is the highest ratio, return the highest ratio
      ASPECT_RATIO_INCR_1=${ASPECT_RATIOS[-1]}
    fi
    # get aspect ratio 2 levels up
    if [[ $((i+2)) -lt ${#ASPECT_RATIOS[@]} ]]; then
      ASPECT_RATIO_INCR_2=${ASPECT_RATIOS[$((i+2))]}
    else
      # If this is the highest ratio, return the highest ratio
      ASPECT_RATIO_INCR_2=${ASPECT_RATIOS[-1]}
    fi
  fi
  i=$((i+1))
done
echo "base aspect ratio: ${BASE_ASPECT_RATIO}"
echo "aspect ratio 1 level up: ${ASPECT_RATIO_INCR_1}"
echo "aspect ratio 2 levels up: ${ASPECT_RATIO_INCR_2}"
echo "Note that this aspect ratio level setting is used for videos only, not images"

echo "NUM_FRAMES=${NUM_FRAMES}"

if [ -z "${NUM_FRAMES}" ]; then
  echo "you need to pass NUM_FRAMES"
else
  let DOUBLE_FRAMES=$2*2
  let QUAD_FRAMES=$2*4
  let OCT_FRAMES=$2*8
fi

echo "DOUBLE_FRAMES=${DOUBLE_FRAMES}"
echo "QUAD_FRAMES=${QUAD_FRAMES}"
echo "OCT_FRAMES=${OCT_FRAMES}"

CMD="python /home/yfeng/ygcheng/src/Open-Sora/scripts/inference.py /home/yfeng/ygcheng/src/Open-Sora/configs/opensora-v1-2/inference/sample.py"
if [[ $CKPT == *"ema"* ]]; then
  parentdir=$(dirname $CKPT)
  CKPT_BASE=$(basename $parentdir)_ema
else
  CKPT_BASE=$(basename $CKPT)
fi
OUTPUT="/home/yfeng/ygcheng/src/Open-Sora/samples/samples_${MODEL_NAME}_${CKPT_BASE}"
start=$(date +%s)
DEFAULT_BS=1

### Functions

# vbench has 950 samples

VBENCH_BS=1 # 80GB
VBENCH_H=240
VBENCH_W=426

# --prompt-as-path
function run_vbench() {
  if [ -z ${VBENCH_RES} ] || [ -z ${VBENCH_ASP_RATIO} ]; then
    eval $CMD --ckpt-path $CKPT --save-dir ${OUTPUT}_vbench --num-sample 5 \
      --prompt-path /home/yfeng/ygcheng/src/Open-Sora/assets/texts/t2v_sora.txt \
      --image-size $VBENCH_H $VBENCH_W \
      --batch-size $VBENCH_BS --num-frames $NUM_FRAMES --start-index $1 --end-index $2
  else
    if [ -z ${NUM_SAMPLING_STEPS} ]; then
        eval $CMD --ckpt-path $CKPT --save-dir ${OUTPUT}_vbench  --num-sample 5 \
        --prompt-path /home/yfeng/ygcheng/src/Open-Sora/assets/texts/t2v_sora.txt \
        --resolution $VBENCH_RES --aspect-ratio $VBENCH_ASP_RATIO \
        --batch-size $VBENCH_BS --num-frames $NUM_FRAMES --start-index $1 --end-index $2
    else
      if [ -z ${FLOW} ]; then
        eval $CMD --ckpt-path $CKPT --save-dir ${OUTPUT}_vbench  --num-sample 5 \
        --prompt-path /home/yfeng/ygcheng/src/Open-Sora/assets/texts/t2v_sora.txt \
        --resolution $VBENCH_RES --aspect-ratio $VBENCH_ASP_RATIO --num-sampling-steps ${NUM_SAMPLING_STEPS} \
        --batch-size $VBENCH_BS --num-frames $NUM_FRAMES --start-index $1 --end-index $2
      else
        if [ -z ${LLM_REFINE} ]; then
          eval $CMD --ckpt-path $CKPT --save-dir ${OUTPUT}_vbench  --num-sample 5 \
          --prompt-path /home/yfeng/ygcheng/src/Open-Sora/assets/texts/t2v_sora.txt \
          --resolution $VBENCH_RES --aspect-ratio $VBENCH_ASP_RATIO --num-sampling-steps ${NUM_SAMPLING_STEPS} --flow ${FLOW} \
          --batch-size $VBENCH_BS --num-frames $NUM_FRAMES --start-index $1 --end-index $2
        else
          if [ "${FLOW}" = "None" ]; then
            eval $CMD --ckpt-path $CKPT --save-dir ${OUTPUT}_vbench  --num-sample 5 \
            --prompt-path /home/yfeng/ygcheng/src/Open-Sora/assets/texts/t2v_sora.txt \
            --resolution $VBENCH_RES --aspect-ratio $VBENCH_ASP_RATIO --num-sampling-steps ${NUM_SAMPLING_STEPS} --llm-refine ${LLM_REFINE} \
            --batch-size $VBENCH_BS --num-frames $NUM_FRAMES --start-index $1 --end-index $2
          else
            eval $CMD --ckpt-path $CKPT --save-dir ${OUTPUT}_vbench  --num-sample 5 \
            --prompt-path /home/yfeng/ygcheng/src/Open-Sora/assets/texts/t2v_sora.txt \
            --resolution $VBENCH_RES --aspect-ratio $VBENCH_ASP_RATIO --num-sampling-steps ${NUM_SAMPLING_STEPS} --flow ${FLOW} --llm-refine ${LLM_REFINE} \
            --batch-size $VBENCH_BS --num-frames $NUM_FRAMES --start-index $1 --end-index $2
          fi
        fi
      fi
    fi
  fi
}

### Main

for arg in "$@"; do
  # image
  if [[ "$arg" = -1 ]] || [[ "$arg" = --image ]]; then
    echo "Running image samples..."
    run_image
  fi
  if [[ "$arg" = -2a ]] || [[ "$arg" = --video ]]; then
    echo "Running video samples a..."
    run_video_a
  fi
  if [[ "$arg" = -2b ]] || [[ "$arg" = --video ]]; then
    echo "Running video samples b..."
    run_video_b
  fi
  if [[ "$arg" = -2c ]] || [[ "$arg" = --video ]]; then
    echo "Running video samples c..."
    run_video_c
  fi
  if [[ "$arg" = -2d ]] || [[ "$arg" = --video ]]; then
    echo "Running video samples d..."
    run_video_d
  fi
  if [[ "$arg" = -2e ]] || [[ "$arg" = --video ]]; then
    echo "Running video samples e..."
    run_video_e
  fi
  if [[ "$arg" = -2f ]] || [[ "$arg" = --video ]]; then
    echo "Running video samples f..."
    run_video_f
  fi
  if [[ "$arg" = -2g ]] || [[ "$arg" = --video ]]; then
    echo "Running video samples g..."
    run_video_g
  fi
  if [[ "$arg" = -2h ]] || [[ "$arg" = --video ]]; then
    echo "Running video samples h..."
    run_video_h
  fi
  # vbench
  if [[ "$arg" = -4 ]] || [[ "$arg" = --vbench ]]; then
    echo "Running vbench samples ..."
    if [ -z ${VBENCH_START_INDEX} ] || [ -z ${VBENCH_END_INDEX} ]; then
      echo "need to set start_index and end_index"
    else
      run_vbench $VBENCH_START_INDEX $VBENCH_END_INDEX
    fi
  fi
  # vbench-i2v
  if [[ "$arg" = -5 ]] || [[ "$arg" = --vbench-i2v ]]; then
    echo "Running vbench-i2v samples ..."
    if [ -z ${VBENCH_START_INDEX} ] || [ -z ${VBENCH_END_INDEX} ]; then
      echo "need to set start_index and end_index"
    else
      run_vbench_i2v $VBENCH_START_INDEX $VBENCH_END_INDEX
    fi
  fi
done

### End

end=$(date +%s)

runtime=$((end - start))

echo "Runtime: $runtime seconds"
