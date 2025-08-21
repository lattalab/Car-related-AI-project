## 🚗 Car Parts and Car Damages
[Dataset link](https://www.kaggle.com/datasets/humansintheloop/car-parts-and-car-damages/code)  
A small testing on Segmentation model. e.g: segformer, YOLOv11-seg

### Car Parts  
Despite the name, this dataset is mainly about car damages.    
I used `YOLOv11n-seg` to train.  

```
For those who are unfamilar with it, you first need to organize your dataset as the following:

dataset
├─ train
│ ├─ img
│ └─ label
├─ val
│ ├─ img
│ └─ label
```
- Each label file should have the same name as its corresponding image.  
- Label content consists of `<class_id>` and polygon points.  

see [document](https://docs.ultralytics.com/datasets/segment/#ultralytics-yolo-format) for more detail.

> Note that YOLO require  `YAML` file when using CIL format training.

### Car Damages  
Actually, This dataset is related to car parts.  
I tried to train `segformer (a transformer-based model)`, but the result are not so much good as I wanted.  

Unlike YOLOv11, segformer require mask data which are much hard to deal with. (maybe only for me)  
- The output of segformer is a segmentation mask.  
- Ground truth masks must be H×W with each pixel representing a class.  (After much tried, the background value set to 255, which will be better than 0)  

The training progress are similar to pytorch workflow.
1. Prepare data  
2. Write dataset and dataloader  
3. Fit the model  
4. Test and evaluate  

Also, some tutorials teach you how to use pytorch-lightning, which provides a clean architecture to speed up development.

### Sound classification
This is another project, click the directory for more detail.