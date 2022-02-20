# Welcome to Salient Imagenet


Salient Imagenet is a dataset for testing the sensitivity of neural networks to different features. To achieve this, we first visualize the top-5 features used by a robust neural network to predict some class and annotate each feature as either core/spurious. Next, we add gaussian noise to the regions containing these features and evaluate the model performance on the noisy images to test the model sensitivity to different features.

## Prerequisites

+ Python >= 3.7
+ Pytorch >= 1.9.0
+ PIL >= 8.4.0 
+ clip. Can be installed using ```pip install git+https://github.com/openai/CLIP.git``` 
+ timm >= 0.4.12
+ pandas >= 1.3.1
+ numpy >= 1.21.2
+ cv2 >= 4.5.3


## Setup

Load the Robust Resnet-50 model using the command given below:   
```wget -O robust_resnet50.pth  https://www.dropbox.com/s/knf4uimlqsi1yz8/imagenet_l2_3_0.pt?dl=0```

## Visualizing class and feature pairs

+ Run visualize_class_feature_pair.ipynb to visualize the feature, class and the Mechanical Turk worker annotations. 
+ Specify ```class_index, feature_index``` in the jupyter notebook to visualize features in Section J of the paper.
+ Example for ```class_index = 325, feature_index = 595``` given below:
![images](./demo_images/325_595_images.jpg)
![heatmaps](./demo_images/325_595_heatmaps.png)
![attacks](./demo_images/325_595_attacks.png)


## Citation

```
@inproceedings{
  singla2022causal,
  title={Salient ImageNet: How to discover spurious features in Deep Learning?},
  author={Sahil Singla and Soheil Feizi},
  booktitle={International Conference on Learning Representations},
  year={2022},
  url={https://openreview.net/forum?id=XVPqLyNxSyh}
}
```
