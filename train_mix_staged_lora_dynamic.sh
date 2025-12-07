#!/bin/bash

export FLUX="black-forest-labs/FLUX.1-dev"
export PANORAMA_DATA="<panorama jsonl path>"
export PERSPECTIVE_DATA="<perspective jsonl path>"
export SAVE_DIR="mix_model_saved"

export CUDA_VISIBLE_DEVICES="0,"
export CUDA_DEVICE=1

python train_mix_staged_lora_dynamic.py \
  --pretrained_model_name_or_path=$FLUX \
  --panorama_data=$PANORAMA_DATA \
  --perspective_data=$PERSPECTIVE_DATA \
  --seed=0 \
  --resolution=1024 \
  --train_batch_size=1 \
  --adam_weight_decay=1e-5 \
  --dataloader_num_workers=25 \
  --save_dir=$SAVE_DIR \
  --devices=$CUDA_DEVICE \
  --max_epochs=25 \
  --guidance_scale=1.0 \
  --rank=64 \
  --lora_alpha=64 \
  --lora_drop_out=0.05 \
  --gaussian_init_lora \
  --precision=16-mixed \
  --accumulate_grad_batches=5 \
  --learning_rate=5e-5 \
  --adam_epsilon=1e-6 \
  --lambda_cube=0.0 \
  --lambda_yaw=0.0 \
  --padding_n=0

  