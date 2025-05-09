# MRED-net
This repository contains the PyTorch implementation of the paper: [**MRED-Net: A Pure Tokens-to-Token Visual Mamba-based Residual Encoder-Decoder Network for Low-Dose CT Denoising**]
***************************************************************************

> **Abstract:**Low-dose computed tomography (LDCT) denoising is crucial for improving image quality and diagnostic accuracy. Effective LDCT denoising models need to incorporate both local structural information and long-range dependencies to achieve good performance. Among the existing methods, convolutional neural networks struggle with capturing long-range dependencies, while Transformers cannot model local information and exhibit quadratic computational complexity. To address these issues, we propose a pure Tokens-to-Token Mamba-based Residual Encoder-Decoder Network (MRED-Net) for LDCT denoising. First, we introduce Mamba, an advanced state space model, as the fundamental component so that our model can capture long-range dependencies while maintaining linear computational complexity. Second, we propose the enhanced Tokens-to-Token (ET2T) blocks to model the local structure represented by surrounding tokens and reduce feature redundancy. Based on the Mamba and ET2T blocks, we construct a residual encoder and decoder architecture to integrate local structure information and long-range dependencies and realize end-to-end image denoising. Moreover, we introduce a compound loss function to enhance anatomical consistency and attain a marked improvement in image quality. In this way, our model incorporates both local information and long-range dependencies, and the latter with linear computational complexity, it achieves thus better image content recovery while removing noise and artifacts. Experiments on clinical and animal data show that MRED-Net attains more appealing results than the state-of-the-art algorithms quantitatively and qualitatively. 
***************************************************************************
### Illustration
<div align=center>
<img src="https://github.com/YuhangLiu98/MRED-net/blob/main/img/MRED-net.png" width="800"/> 
</div>

-------

### DATASET

1.The Mayo Clinic Low Dose CT by Mayo Clinic   
(I can't share this data, you should ask at the URL below if you want)  
https://www.cancerimagingarchive.net/collection/ldct-and-projection-data/

2.The Piglet Low Dose CT by X. Yi and P. Babyn, "Sharpness-aware low-dose CT denoising using conditional generative adversarial network"
(I can't share this data, you should ask at the URL below if you want)  
https://github.com/xinario/SAGAN?tab=readme-ov-file


-------
## Installation

MRED-net can be installed from source,
```shell
git clone https://github.com/YuhangLiu98/MRED-net.git
```
Then, [Pytorch](https://pytorch.org/) is required, for example,
```shell script
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
Lastly, other pakages are required,
```shell script
pip install -r requirements.txt
```

-------

## Use

1. run `python prep.py` to convert 'dicom file' to 'numpy array'
2. run `python train.py` to start training.
3. run `python test.py` to start testing.
-------

### RESULT  
<div align=center>
<img src="https://github.com/YuhangLiu98/MRED-net/blob/main/img/result1.png" width="800"/>   
<img src="https://github.com/YuhangLiu98/MRED-net/blob/main/img/result2.png" width="800"/>   
<img src="https://github.com/YuhangLiu98/MRED-net/blob/main/img/result4.png" width="800"/>   
</div>
