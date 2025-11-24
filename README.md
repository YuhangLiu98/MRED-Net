# 🐍 MRED-Net: A pure tokens-to-token visual mamba-based residual encoder-decoder network for low-dose CT denoising [![paper](https://img.shields.io/badge/ESWA-Journal%20Paper-blue)](https://www.sciencedirect.com/science/article/pii/S095741742503920X?dgcid=coauthor) 

> **Abstract:** Low-dose computed tomography (LDCT) denoising is crucial for improving image quality and diagnostic accuracy. Effective LDCT denoising models need to incorporate both local structural information and long-range dependencies to achieve good performance. Among the existing methods, convolutional neural networks struggle with capturing long-range dependencies, while Transformers cannot model local information and exhibit quadratic computational complexity. To address these issues, we propose a pure Tokens-to-Token Mamba-based Residual Encoder-Decoder Network (MRED-Net) for LDCT denoising. First, we introduce Mamba, an advanced state space model, as the fundamental component so that our model can capture long-range dependencies while maintaining linear computational complexity. Second, we propose the enhanced Tokens-to-Token (ET2T) blocks to model the local structure represented by surrounding tokens and reduce feature redundancy. Based on the Mamba and ET2T blocks, we construct a residual encoder and decoder architecture to integrate local structure information and long-range dependencies and realize end-to-end image denoising. Moreover, we introduce a compound loss function to enhance anatomical consistency and attain a marked improvement in image quality. In this way, our model incorporates both local information and long-range dependencies, and the latter with linear computational complexity, it achieves thus better image content recovery while removing noise and artifacts. Experiments on clinical and animal data show that MRED-Net attains more appealing results than the state-of-the-art algorithms quantitatively and qualitatively.
## 📅 Illustration
<div align=center>
<img src="https://github.com/YuhangLiu98/MRED-net/blob/main/img/MRED-net.png" width="800"/> 
</div>

## 🛠️ Installation
1. MRED-net can be installed from source,
```shell
git clone https://github.com/YuhangLiu98/MRED-net.git
```
2. Then, [Pytorch](https://pytorch.org/) is required, for example,
```shell script
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
3. Additionally, Mamba dependencies are required,
please refer to [VM-UNet](https://github.com/JCruan519/VM-UNet).

## 💽 Dataset Download
1. The Mayo Clinic Low Dose CT by Mayo Clinic   
(I can't share this data, you should ask at the URL below if you want)  
https://www.cancerimagingarchive.net/collection/ldct-and-projection-data/

2. The Piglet Low Dose CT by X. Yi and P. Babyn, "Sharpness-aware low-dose CT denoising using conditional generative adversarial network"
(I can't share this data, you should ask at the URL below if you want)  
https://github.com/xinario/SAGAN?tab=readme-ov-file


## 🗃️ Data Preparation:
Run `python prep.py` to convert 'dicom file' to 'numpy array', please refer to https://github.com/SSinyu/RED-CNN for more detailed data preparation. 
The path of .npy files for training and testing can be set as
```
python main.py --train_data_path [your_train_path] --test_data_path [your_test_path]
```

## 🔥 Model Training and Testing:
```
>> python main.py  ## train CTformer. 
>> python main.py --mode test --test_iters [set iters]  ## run test.
```

## 📊 Result 
<div align=center>
<img src="https://github.com/YuhangLiu98/MRED-net/blob/main/img/result3.png" width="600"/>   
<img src="https://github.com/YuhangLiu98/MRED-net/blob/main/img/result1.png" width="1000"/>   </div>


## ✍️ Citation
```shell
@article{LIU2026130305,
author = {Yuhang Liu and Huazhong Shu and Qiang Chi and Yue Zhang and Zidong Liu and Fuzhi Wu and Xin Chen and Lei Wang and Yi Liu and Pengcheng Zhang and Zhiguo Gui},
title = {MRED-Net: A pure tokens-to-token visual mamba-based residual encoder-decoder network for low-dose CT denoising},
journal = {Expert Systems with Applications},
volume = {301},
pages = {130305},
year = {2026},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2025.130305}
}
```
## 🤝 Acknowledgment
Part of our code is based on the [VM-UNet](https://github.com/JCruan519/VM-UNet) and [CTformer](https://github.com/wdayang/ctformer).
Thanks for their awesome work.
