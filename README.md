# VI-GAN
An unofficial implementation of [View Independent Generative Adversarial Network for Novel View Synthesis](https://openaccess.thecvf.com/content_ICCV_2019/papers/Xu_View_Independent_Generative_Adversarial_Network_for_Novel_View_Synthesis_ICCV_2019_paper.pdf).

![Architecture](/image/architecture.PNG)
|Target|![](/image/1_target.png) |![](/image/2_target.png)|![](/image/3_target.png) 
|:---:|:---:|:---:|:---:|
|Predict|![](/image/1_predict.png) |![](/image/2_predict.png)|![](/image/3_predict.png) 


## Getting Started
### Dependencies
* pytorch 1.5.1 
* torchvision 0.6.1 

### Train
```
python train.py --train_dir [/folder/to/training/dataset] --batch_size [] --checkpoint [./checkpoint/0.pth]
```
### Test
```
python test.py --test_dir [/folder/to/testing/dataset] --checkpoint [./checkpoint/x.pth]
```

## Data
The data is based on the Shapenet dataset naming convention: {model name}\_{azimuth/10}\_{elevation}.png

## Reference
[1] Xu, Xiaogang, Ying-Cong Chen, and Jiaya Jia. "View Independent Generative Adversarial Network for Novel View Synthesis." Proceedings of the IEEE International Conference on Computer Vision. 2019.
