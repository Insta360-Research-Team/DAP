from functools import partial
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxPipeline, FluxTransformer2DModel
from peft import LoraConfig
import diffusers
from datasets import load_dataset
from lightning.pytorch.strategies import DDPStrategy
from pytorch_lightning import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
import lightning as L
from torchvision import transforms
from torch.utils.data import random_split
from torch import nn
import argparse
import os
import gc
import copy
import torch
import random

from src.data import prepare_train_dataset
from src.pipeline import DiT360Pipeline
from src.dit360 import DiT360


def collate_fn(examples, text_encoding_pipeline):
    pixel_values = torch.stack([example["pixel_values"]
                               for example in examples])
    pixel_values = pixel_values.to(
        memory_format=torch.contiguous_format).float()
    captions = [example["captions"] for example in examples]

    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = text_encoding_pipeline.encode_prompt(
            captions, prompt_2=None)

    return {
        "pixel_values": pixel_values,
        "prompt_embeds": prompt_embeds,
        "pooled_prompt_embeds": pooled_prompt_embeds,
        "text_ids": text_ids
    }


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="black-forest-labs/FLUX.1-dev",
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="A seed for reproducible training."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9,
                        help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999,
                        help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float,
                        default=0, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-06,
                        help="Epsilon value for the Adam optimizer")
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help="The guidance scale used for transformer.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from."
    )
    parser.add_argument(
        "--load_ckpt",
        type=str,
        default=None,
        help="Path for transformer backbone."
    )
    ########################################
    # for lora setting
    parser.add_argument(
        "--rank",
        type=int,
        default=64,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=64
    )
    parser.add_argument(
        "--lora_bias",
        type=str,
        default="none",
        help="If training the bias of lora_B layers."
    )
    parser.add_argument(
        "--gaussian_init_lora",
        action="store_true",
        help="If using the Gaussian init strategy. When False, we follow the original LoRA init strategy.",
    )
    parser.add_argument(
        "--lora_drop_out",
        type=float,
        default=0.05,
    )
    ########################################
    parser.add_argument(
        "--resolution", 
        type=int, 
        default=1024,
        help="The height of the generated panorama image; the width is automatically set to twice this value."
    )
    parser.add_argument("--train_batch_size", type=int, default=10)
    parser.add_argument("--dataloader_num_workers", type=int, default=24)
    parser.add_argument("--save_dir", type=str,
                        help="The folder path to save the checkpoint.")
    parser.add_argument("--topk", type=int, default=-1,
                        help="How many the top ckpt to be saved.")
    parser.add_argument("--devices", type=str, default="0",
                        help="The GPUs used to train the model.")
    parser.add_argument("--max_epochs", type=int, default=50,
                        help="The epochs to train the model. One epoch means use all data in dataloader once.")
    parser.add_argument("--precision", type=str, default="16",
                        help="Mixed precision training.")
    parser.add_argument("--accumulate_grad_batches", type=int,
                        default=1, help="Accumulate grad batches.")

    ########################################
    parser.add_argument(
        "--lambda_cube", 
        type=float,
        default=0.50,
        help="The coefficient of the cube loss."
    )
    parser.add_argument(
        "--lambda_yaw", 
        type=float,
        default=0.50,
        help="The coefficient of the yaw loss."
    )
    parser.add_argument(
        "--padding_n", 
        type=int,
        default=1,
        help="The numbers of padding columns."
    )

    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    if args.seed is not None:
        seed_everything(args.seed)

    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)

    target_modules = [
        "attn.to_k",
        "attn.to_q",
        "attn.to_v",
        "attn.to_out.0",
    ]
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights="gaussian" if args.gaussian_init_lora else True,
        target_modules=target_modules,
        bias=args.lora_bias,
        lora_dropout=args.lora_drop_out
    )

    # Prepare dataset and dataloader.
    train_dataset = load_dataset("Insta360-Research/Matterport3D_polished")
    train_dataset = prepare_train_dataset(train_dataset, args.resolution)

    text_encoding_pipeline = DiT360Pipeline.from_pretrained(
        args.pretrained_model_name_or_path, vae=None, transformer=None)

    collate = partial(
        collate_fn, text_encoding_pipeline=text_encoding_pipeline)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        drop_last=True,
        shuffle=True,
        collate_fn=collate,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=args.topk,
        every_n_epochs=1,
        filename="vsclip_{epoch:02d}_{train_loss:.2f}",
        save_last=True
    )

    trainer = L.Trainer(
        devices=args.devices,
        accelerator="gpu",
        callbacks=[checkpoint_callback],
        default_root_dir=args.save_dir,
        max_epochs=args.max_epochs,
        precision=args.precision,
        strategy='deepspeed_stage_2',
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=100
    )

    # load model
    args.max_steps = len(train_dataloader) * args.max_epochs // args.accumulate_grad_batches
    model = DiT360(args, lora_config=transformer_lora_config, ckpt=args.load_ckpt)

    trainer.fit(model, train_dataloader, ckpt_path=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
