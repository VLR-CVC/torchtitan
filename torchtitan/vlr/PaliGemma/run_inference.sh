#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

MODEL_PATH='/data/users/mserrao/.cache/huggingface/hub/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c'
PROMPT='Detect all dogs in the image and provide Bounding Boxes: '
IMAGE_PATH='/home-local/mserrao/PaliGemma/images/image2.jpg'
MAX_TOKENS_TO_GENERATE=2048
TEMPERATURE=0.5
TOP_P=0.9
DO_SAMPLE=True
ONLY_CPU=False

python inference.py \
    --model_path "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --image_file_path "$IMAGE_PATH" \
    --max_tokens_to_generate $MAX_TOKENS_TO_GENERATE \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --do_sample $DO_SAMPLE \
    --only_cpu $ONLY_CPU