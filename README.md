# ForgeLens: Data-Efficient Forgery Focus for Generalizable Forgery Image Detection [ICCV 2025]
[![Static Badge](https://img.shields.io/badge/2408.13697-red?style=flat&logo=arxiv&logoColor=%23B31B1B&label=Arxiv&labelColor=%23FFFFFF&color=%23B31B1B&link=https%3A%2F%2Farxiv.org%2Fpdf%2F2408.13697)](https://arxiv.org/abs/2408.13697)

**ForgeLens** is a data-efficient, feature-guided CLIP-ViT framework for detecting AI-generated images with strong generalization to unseen forgery techniques. It guides the frozen CLIP-ViT to focus on forgery-relevant information within the general-purpose features it extracts, addressing a key limitation of prior frozen-network-based methods, which often retain excessive forgery-irrelevant content. ForgeLens introduces two simple and lightweight modules—WSGM and FAFormer—to guide the model’s attention toward forgery-specific cues. With only 1% of the training data, ForgeLens outperforms existing forgery detection methods.

<p align="center">
  <img src="Figs/forgelens.png" style="max-width:100%; height:auto;">
</p>

## ⚙️ Environment Setup
To ensure reproducibility and avoid compatibility issues, we recommend setting up the required environment by installing the necessary packages using the following command:
```
pip install -r requirements.txt
```
## 📦 Get the Required Dataset
**Training dataset**
we utilize ProGAN training set to train our network following [CNNDetection](https://github.com/peterwang512/CNNDetection).

**Evaluation dataset**
In order to fully evaluate ForgeLens, we conducted tests on the [UniversalFakeDetect](https://github.com/WisconsinAIVision/UniversalFakeDetect) dataset.

🔽 Preprocessed Version (Recommended)
You can also directly download our pre-split dataset, which includes three training settings:

- 1️⃣Setting-**1** (1,600 samples)
- 2️⃣Setting-**2** (50% of the full training set)
- 3️⃣Setting-**3** (100% of the full training set)

👉 📁 [Download Link](https://pan.baidu.com/s/11CHrO8KpiJYi8SeQfuIxoA?pwd=xr4s).

## 📂 Dataset Split

We adopt a binary classification setting for all datasets, where:  
- **0**: real image  
- **1**: fake/generated image

### Training Set

We use a subset of **ProGAN** data for training:
```
Training/
├── car/
│   ├── 0_real/        # Real images
│   └── 1_fake/        # ProGAN-generated fake images
├── cat/
│   ├── 0_real/        
│   └── 1_fake/        
├── chair/
│   ├── 0_real/        
│   └── 1_fake/        
└── horse/
    ├── 0_real/        
    └── 1_fake/        
```

### Evaluation Set

Our method is evaluated on ***UniversalFakeDetect***, using the same folder structure as the training set.  

## 🛠️ Configuration
You can manage the hyperparameters in ``./options/options.py``.

## 🧠 Training
You can perform training using the following command:
```
bash train_setting_1.sh   # reproduce the results of training setting 1
bash train_setting_2.sh   
bash train_setting_3.sh   
```

## 🚀 Quick Reproducing
We provide pretrained weights corresponding to the results reported in our paper.  
### 📦 Pretrained Weights
| Training Setting     | Description                 |                 Google Drive Link                  |                            Baidu Netdisk Link                            |
|----------------------|-----------------------------|:--------------------------------------------------:|:------------------------------------------------------------------------:|
| `training_setting_1` | 1,600 samples (Recommended) |     [Download](https://drive.google.com/file/d/1JxfFqVrX50U5FFR_Wm1BGYVtIi-IX_sH/view?usp=sharing)     |   [Download](https://pan.baidu.com/s/15l_lzgvb6nAF8z6u7T9fuw?pwd=6fyb)   |
| `training_setting_2` | 50% of full training data   |     [Download](https://drive.google.com/file/d/1DBxVW0Z0_EPcjt7vdQPea92mf6Zk6YBc/view?usp=sharing)     |   [Download](https://pan.baidu.com/s/1uiHLUWnX8d-KRviX77j5cw?pwd=wkdd)   |
| `training_setting_3` | 100% of full training data  |     [Download](https://drive.google.com/file/d/1lhIri-prWLbg9uAg0XqXtD8nzKrFIWlh/view?usp=sharing)     |                [Download](https://pan.baidu.com/s/1-DLRQaqp5VW0bxfh1HmvdA?pwd=ef4n)                 |

To reproduce the results of `Table 1` and `Table 2` in our paper, first download the pretrained weights, then run the evaluation script as follows:
```bash
bash evaluate.sh
```

## 🔄 Update (2025-09-15): Evaluation on GenImage  

### 1. Dataset Download  
Download the **GenImage dataset** from the [official source](https://genimage-dataset.github.io/) or download [our pre-split dataset](https://pan.baidu.com/s/19MIJccFZiHGIIsF18Rozkw?pwd=895s).

### 2. Quick Evaluation with Our Provided Checkpoint 
First download the pretrained weights from [Google Drive](https://drive.google.com/file/d/1ZGmFPDJFtiJBxe3_mQGP_ZKPO_XXzjWc/view?usp=sharing) of [Baidu Netdisk](https://pan.baidu.com/s/1H50DjdnnnfDJqn8O75sa4g?pwd=sv35), then run the evaluation script as follows:
```bash
bash evaluate_GenImage.sh
```

### 3. Training
You can perform training using the following command:
```
bash train_GenImage.sh
```
## 📌 Citation

If you find our project helpful, please feel free to leave a ⭐ and cite our paper:

```bibtex
@InProceedings{Chen_2025_ICCV,
    author    = {Chen, Yingjian and Zhang, Lei and Niu, Yakun},
    title     = {ForgeLens: Data-Efficient Forgery Focus for Generalizable Forgery Image Detection},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {16270-16280}
}
