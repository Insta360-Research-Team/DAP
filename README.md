# DiT360: High-Fidelity Panoramic Image Generation via Hybrid Training

<p align="center">
  <a href='https://arxiv.org/abs/2510.11712'><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>
  <a href='https://fenghora.github.io/DiT360-Page/'><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=insta360&logoColor=white' alt='Project Page'></a>
  <a href='https://huggingface.co/spaces/Insta360-Research/DiT360'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live_Demo-blue'></a>
  <a href='https://huggingface.co/datasets/Insta360-Research/Matterport3D_polished'><img src='https://img.shields.io/badge/%F0%9F%93%88%20Hugging%20Face-Dataset-yellow'></a>
</p>

![teaser](assets/teaser.jpg)

**DiT360** is a framework for high-quality panoramic image generation, leveraging both **perspective** and **panoramic** data in a hybrid training scheme.
It adopts a two-level strategy—**image-level cross-domain guidance** and **token-level hybrid supervision**—to enhance perceptual realism and geometric fidelity.

## 🌟 Features

<!-- <p align="center">
  <img src="assets/result.gif" width="90%">
</p> -->
![result](assets/result.gif)

- **High Perceptual Realism**: It produces panoramic images with high resolution and clear details for lifelike visual quality.
- **Precise Geometric Fidelity**: It correctly models multi-scale distortions in panoramic images with smooth, continuous edges.
- **Versatile Applications**: It enables robust handling of generated assets across multiple tasks, seamlessly supporting both inpainting and outpainting.


## ⏩ Updates

**30/10/2025**
- Release mix-training code.

**17/10/2025**
- Release the inpainting and outpainting code.

**15/10/2025**
- Release the training code.

**14/10/2025**
- Release our paper on arXiv.

**11/10/2025**
- Release the refined Matterport3D dataset.
  
**10/10/2025**
- Release the pretrained model and inference code.

## 🔨 Installation

Clone the repo first:

```Bash
git clone https://github.com/Insta360-Research-Team/DiT360.git
cd DiT360
```

(Optional) Create a fresh conda env:

```Bash
conda create -n dit360 python=3.12
conda activate dit360
```

Install necessary packages (torch > 2):

```Bash
# pytorch (select correct CUDA version, we test our code on torch==2.6.0 and torchvision==0.21.0)
pip install torch==2.6.0 torchvision==0.21.0

# other dependencies
pip install -r requirements.txt
```

## 🖼️ Dataset

We have uploaded the dataset to **Hugging Face**. For more details, please visit [Insta360-Research/Matterport3D_polished](https://huggingface.co/datasets/Insta360-Research/Matterport3D_polished).

For a quick start, you can try:

```python
from datasets import load_dataset

ds = load_dataset("Insta360-Research/Matterport3D_polished")

# check the data
print(ds["train"][0])
```
If you encounter any issues, please refer to the official Hugging Face documentation.


## 📒 Inference

For a quick use, you can just try:

```Bash
python inference.py
```

⚠️ Note: We only trained the model on datasets with a resolution of **1024 × 2048**, so using other input sizes may lead to abnormal results.
In addition, without any optimization, the inference process requires approximately **37 GB** of GPU memory, so please be aware of the memory usage.

## 🚀 Train

### Panorama Training

We provide a training pipeline based on **Insta360-Research/Matterport3D_polished**, along with corresponding launch scripts.
You can start training with a single command:


```bash
bash train.sh
```


After training is completed, you will find a checkpoint file saved under the output directory, typically like:


```bash
model_saved/lightning_logs/version_x/checkpoints/vsclip_epoch=xxx.ckpt/checkpoint/mp_rank_00_model_states.pt
```


You can extract the LoRA weights from the full `.pt` checkpoint by running:


```bash
python get_lora_weights.py <path_to_your_pt_file> <output_dir>
```


If you don’t specify `output_dir`, the extracted weights will be saved by default to:


```bash
lora_output/
```


After that, you can directly use your trained LoRA in the inference script.
Simply replace the default model path `"fenghora/DiT360-Panorama-Image-Generation"` in `inference.py` with your output directory (e.g., `"lora_output"`), and then run:


```bash
python inference.py
```

### Mix Training


Mix training aims to leverage both **panoramic images** and **perspective images** to improve the model’s generalization across different viewpoints.  

#### Data Preparation

You need to prepare **two `.jsonl` files**:  
- One for **panoramic images**  
- One for **perspective images**

Each line in a `.jsonl` file should represent a single training sample with the following format:

```json
{"image": "path/to/image.jpg", "caption": "a description of the scene", "mask": "path/to/mask.png"}
```

#### Mask Description

The `mask` is a PNG (or similar) image used to specify which regions should be supervised during training:

- **White regions (`255, 255, 255`)** indicate areas that are supervised.  
- **Black regions (`0, 0, 0`)** indicate areas that are ignored.

Specifically:

- For **panoramic images**, the `mask` is typically an all-white image (meaning the entire image is supervised).  
- For **perspective images**, the `mask` corresponds to the valid projected area derived from the panoramic-to-perspective mapping.

#### Projection Mapping

The perspective images and their corresponding masks can be generated from panoramic images using an equirectangular-to-perspective projection.  
We highly recommend using the excellent open-source library below for this purpose:  

> [py360convert](https://github.com/sunset1995/py360convert)

This library provides high-quality conversions between panoramic and perspective views, making it easy to generate consistent training data for mixed-view learning.

To start training, please refer to the provided scripts:  
`train_mix_staged_lora_dynamic.sh` and `train_mix_staged_lora_dynamic.py`.

## 🎨 Inpainting & Outpainting


We treat both **inpainting** and **outpainting** as image completion tasks, where the key lies in how the **mask** is defined. A simple example is already provided in our codebase.

For a quick start, you can simply run:
```bash
python editing.py
```

In our implementation, regions with a mask value of 1 correspond to the parts preserved from the source image. 
Therefore, in our example, you can invert the mask as follows for inpainting:
```python
mask = 1 - mask  # for inpainting
```

This part is built upon [Personalize Anything](https://github.com/fenghora/personalize-anything).


## 🤝 Acknowledgement

We appreciate the open source of the following projects:

* [diffusers](https://github.com/huggingface/diffusers)
* [Personalize Anything](https://github.com/fenghora/personalize-anything)
* [RF-Inversion](https://github.com/LituRout/RF-Inversion)
* [RF-Solver-Edit](https://github.com/wangjiangshan0725/RF-Solver-Edit)

## Citation
```
@misc{dit360,
  title={DiT360: High-Fidelity Panoramic Image Generation via Hybrid Training}, 
  author={Haoran Feng and Dizhe Zhang and Xiangtai Li and Bo Du and Lu Qi},
  year={2025},
  eprint={2510.11712},
  archivePrefix={arXiv},
}
```
If you find our **dataset** useful, please include a citation for **Matterport3D**:
```
@article{Matterport3D,
  title={Matterport3D: Learning from RGB-D Data in Indoor Environments},
  author={Chang, Angel and Dai, Angela and Funkhouser, Thomas and Halber, Maciej and Niessner, Matthias and Savva, Manolis and Song, Shuran and Zeng, Andy and Zhang, Yinda},
  journal={International Conference on 3D Vision (3DV)},
  year={2017}
}
```
If you find our **inpainting & outpainting** useful, please include a citation for **Personalize Anything**:
```
@article{feng2025personalize,
  title={Personalize Anything for Free with Diffusion Transformer},
  author={Feng, Haoran and Huang, Zehuan and Li, Lin and Lv, Hairong and Sheng, Lu},
  journal={arXiv preprint arXiv:2503.12590},
  year={2025}
}
```