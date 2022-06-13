# DGP(Deep Generative Projection) - Official Tensorflow Implementation of CVPR2022 Paper Weakly Supervised High-Fidelity Clothing Model Generation

> ## Weakly Supervised High-Fidelity Clothing Model Generation
> In CVPR 2022<br>
> [[Project Page]](https://ruilifeng.github.io/Deep-Generative-Projection/)  [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Feng_Weakly_Supervised_High-Fidelity_Clothing_Model_Generation_CVPR_2022_paper.pdf)  [[Supp]](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Feng_Weakly_Supervised_High-Fidelity_CVPR_2022_supplemental.pdf)  [[arxiv]](https://arxiv.org/pdf/2112.07200.pdf)<br>
>![Clothing Model Generation](imgs/figure1.png "Clothing Model Generation")<br>
> **Abstract**: The development of online economics arouses the demand of generating images of models on product clothes, to display new clothes and promote sales. However, the expensive  proprietary model images challenge the existing image virtual try-on methods in this scenario, as most of them need to be trained on considerable amounts of model images accompanied with paired clothes images. In this paper, we propose a cheap yet scalable weakly-supervised method called Deep Generative Projection (DGP) to address this specific scenario. Lying in the heart of the proposed method is to imitate the process of human predicting the wearing effect, which is an unsupervised imagination based on life experience rather than computation rules learned from supervisions. Here a pretrained StyleGAN is used to capture the practical experience of wearing. Experiments show that projecting the rough alignment of clothing and body onto the StyleGAN space can yield photo-realistic wearing results. Experiments on real scene proprietary model images demonstrate the superiority of DGP over several state-of-the-art supervised methods when generating clothing model images.

## Installation
Requirements: TensorFlow 1.14, CUDA 10.0, cuDNN 7.5
```
conda create -n DGP python=3.6 cudatoolkit=10.0
conda activate DGP
pip install -r requirements.txt
```

## Dataset
### The E-Shop Fashion (ESF) Dataset
- The ESF dataset now can be downloaded from [here](https://mailustceducn-my.sharepoint.com/:f:/g/personal/frl1996_mail_ustc_edu_cn/EpNhuXT-hS5FstHNlv3TeJQBeJYzG0hKkPUgeZ-IO9-zww?e=DBYohK)
- The E-Shop Fashion (ESF) dataset consists of 173,567 clothing model images from the Internet. The images are all cropped to the region between jaw and thigh, and resized to the resolution of 512 × 512. 
- ![Examples of the ESF dataset](imgs/ESF.png "Examples of the ESF dataset")
- ![Number of each category on the ESF dataset](imgs/sta_cloth.png "Number of each category on the ESF dataset")


### The Commercial Model Image (CMI) Dataset
- The CMI dataset cannot be released completely according to the copyright and the privacy issues. However we are trying to offer a small part for testing. As the CMI dataset is only a benchmark for evaluating our method, users may choose other public datasets (VTON, DeepFashion) for test.

## Dataset Preprocess
The data are organized as follows. Note each model contains a **model_info directory**, which contains cropped model image and corresponding parsing info. Each model has multiple **cloth directories**, which contains cloth img, cloth parse, coarse aligned img, and the resudial mask.

```
data/
  |_model_1/
    |_model_info/
      |_model_img.png
      |_model_parse.png
    |_cloth_1/
      |_cloth_parse.png
      |_coarse_align.png
      |_residual_parse.png
    |_cloth_2/
      ...
  |_model_2/
    ...
```

We will release the related models and prepsocess scripts to get preprocessed data soon.

TODO List: 
- [x] Release the ESF dataset
- [ ] Release human parsing / keypoints model
- [ ] Release the clothing parsing / keypoints model
- [ ] Release the coarse alignment / crop script


## Pretrained models
download the [pretrained models zip file](https://mailustceducn-my.sharepoint.com/:u:/g/personal/frl1996_mail_ustc_edu_cn/ESTZWdXXiaxJuvr7ihrDRGQBbMYNhNkMI2R9nmqBk99HQg?e=ToSae6) and unzip to the current directory

```
unzip pretrained_models.zip
```

you will get the following models in `pretrained_models/`:
- encoder-stylegan2-upper-body-512.pkl ( stylegan2 model with encoder, trained on ESF upper body dataset)
- vgg.h5 ( for calculating the perceptual loss )
- attributes_model_resnet101.h5 ( for calculating the attribute loss)


## Run the demo
```
python deepGenPro.py --cloth_dir=./data/model_1/cloth_1/ --model_dir=./data/model_1/model_info/ --cloth_sleeve=short --output_dir=./output/
```

the results are in the save file:

![Clothing Model Generation](imgs/show_case.png "Clothing Model Generation")<br>


## Docker usage

  - Build
  ```
  docker build --rm -t dgp .
  ```
  
  - Inference
  ```
  docker run --name DGP --gpus all --rm -it -v <absolute_local_path>:/workspace/ dgp   --cloth_dir=./data/model_1/cloth_1/ --model_dir=./data/model_1/model_info/ --cloth_sleeve=short --output_dir=./output/
  ```


## BibTeX
```
@InProceedings{Feng_2022_CVPR,
    author    = {Feng, Ruili and Ma, Cheng and Shen, Chengji and Gao, Xin and Liu, Zhenjiang and Li, Xiaobo and Ou, Kairi and Zhao, Deli and Zha, Zheng-Jun},
    title     = {Weakly Supervised High-Fidelity Clothing Model Generation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {3440-3449}
}
```