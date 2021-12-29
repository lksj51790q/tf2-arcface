# ArcFace in TensorFlow 2
### Introduction
<a href="https://openaccess.thecvf.com/content_CVPR_2019/html/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.html">ArcFace</a>
is the state-of-the-art model for face recognition. It enhances the intra-class compactness and inter-class discrepancy to solve the open-set problem. The original development is in MXNet by <a href="https://github.com/deepinsight/insightface">InsightFace</a>.<br>
### Dataset Preparing
The VGGface2 dataset is used. It contains 3,311,286 images and consists of 9,131 identities.
1. Download dataset:&emsp;<a href="https://www.robots.ox.ac.uk/~vgg/data/vgg_face/">VGGFace2</a><br>
2. Cropping only face and resize images to 112 x 112
### Training and Validation
The validation dataset is randomly split 5% from all data by image_dataset_from_directory(). The training program can distribute training across multiple GPUs.
```
python train.py
```
### Result
Validation Accuracy: 96.56514%<br>
Download Model: <a href="https://drive.google.com/file/d/1Mpg9ALfbQssK3c59e8m6ciGYoFbFcyyl/view?usp=sharing">Google Drive</a>
