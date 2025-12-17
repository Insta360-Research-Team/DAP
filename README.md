# Depth Any Panoramas: A Foundation Model for Panoramic Depth Estimation

### Xin Lin, Meixi Song, Dizhe Zhang, Wenxuan Lu, Haodong Li, Bo Du, Ming-Hsuan Yang, Truong Nguyen, Lu Qi


[**Xin Lin**](https://linxin0.github.io) · [**Meixi Song**] · [**Dizhe Zhang**] · [**Wenxuan Lu**] · [**Haodong Li**](https://haodong2000.github.io)
<br>
[**Bo Du**] · [**Ming-Hsuan Yang**] · [**Truong Nguyen**] · [**Lu Qi**](http://luqi.info)

<p align="center">
  <a href=''><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>
  <a href='https://insta360-research-team.github.io/DAP_website/'><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=insta360&logoColor=white' alt='Project Page'></a>
  <a href=''><img src='https://img.shields.io/badge/%F0%9F%93%88%20Hugging%20Face-Dataset-yellow'></a>
  <a href='https://huggingface.co/spaces/Insta360-Research/DAP'><img src='https://img.shields.io/badge/🚀%20Hugging%20Face-Demo-orange'></a>
</p>

![teaser](assets/depth_teaser2_00.png)



## 🔨 Installation

Clone the repo first:

```Bash
git clone https://github.com/Insta360-Research-Team/DAP
cd DAP
```

(Optional) Create a fresh conda env:

```Bash
conda create -n dap python=3.12
conda activate dap
```

Install necessary packages (torch > 2):

```Bash
# pytorch (select correct CUDA version, we test our code on torch==2.7.1 and torchvision==0.22.1)
pip install torch==2.7.1 torchvision==0.22.1

# other dependencies
pip install -r requirements.txt
```

## 🖼️ Dataset

The training dataset will be open soon.


## 🤝 Pre-trained model

Please download the pretrained model: https://huggingface.co/Insta360-Research/DAP-weights


## 📒 Inference

```Bash
python test/infer.py 
```


## 🚀 Evaluation


```Bash
python test/eval.py 
```




## 🤝 Acknowledgement

We appreciate the open source of the following projects:

* [PanDA](https://caozidong.github.io/PanDA_Depth/)
* [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)


## Citation
```
@article{lin2025dap,
          title={Depth Any Panoramas: A Foundation Model for Panoramic Depth Estimation},
          author={Lin, Xin and Song, Meixi and Zhang, Dizhe and Lu, Wenxuan and Li, Haodong and Du, Bo and Yang, Ming-Hsuan and Nguyen, Truong and Qi, Lu},
          journal={arXiv},
          year={2025}
        }
```

