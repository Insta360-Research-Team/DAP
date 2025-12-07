# First stage: Fine-tune a teacher model using synthetic labeled data

We provide a codebase to fine-tune the Depth Anything v2 model using synthetic labeled data. In our practice, the ViT encoder is frozen with LoRA layers added, while the DPT decoder is fully trained. We utilize an indoor synthetic dataset (Structured3D) and an outdoor synthetic dataset (Deep360).

## Usage

### Checkpoint Prepraration

Download the checkpoints of Depth Anything v2 [here](https://github.com/DepthAnything/Depth-Anything-V2) and put them under the `checkpoints` directory.

### Dataset Prepraration

Download the [Structured3D](https://structured3d-dataset.org/) and [Deep360](https://github.com/nju-ee/MODE-2022) datasets.

Then, modify the dataset path in the corresponding yaml files.

```bash
./config/teacher/train.yaml
```

### Training

We take the `vitl` encoder as an example. You can also use `vitb` or `vits` encoders by modifying the model config in the `train.yaml` file.

*`Note`*: A NVIDIA 3090 GPU is enough to train with `vits` encoder. Instead, `vitl` might require larger memory, such as NVIDIA A800 GPU.

```bash
python train_teacher/joint_train.py \
       --config ./config/teacher/train.yaml
```

### Evaluation

After training on synthetic datasets, we evaluate on two real-world datasets (Matterport3D and Stanford2D3D). The dataset preparation can refer to the folder [train_metric_depth](../train_metric_depth/).

```bash
python train_teacher/eval_multi_dataset.py \
       --config ./config/teacher/test.yaml 
```