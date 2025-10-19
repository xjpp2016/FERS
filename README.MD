# Rebellious Student: A Complementary Distillation Framework for Background Feature Enhancement in Hyperspectral Anomaly Detection
**Authors**: Wenping Jin, Yuyang Tang, Li Zhu

## Overview
This repository contains the implementation for the second-stage training (i.e., training of the spatial branch based on complementary distillation) and the final fusion detection scheme presented in our paper: 

**"Rebellious Student: A Complementary Distillation Framework for Background Feature Enhancement in Hyperspectral Anomaly Detection"**

The first-stage training pipeline in our paper builds upon our previous work, **FERD**. The corresponding code is available at: https://github.com/xjpp2016/FERD

## Abstract
Most hyperspectral anomaly detection .

![image](https://github.com/cristianoKaKa/FERD/blob/master/framework.png)

## Performance on HAD100 Dataset

### Dataset

HAD100: https://zhaoxuli123.github.io/HAD100/

### Running Tests

To run the tests on HAD100, use the following command:

- For first 50 bands test:
```bash
python test.py --input_channel 50
```
- For first 100 bands test:
```bash
python test.py --input_channel 100
```
- For first 200 bands test:
```bash
python test.py --input_channel 200
```

### Running Training on HAD100

```bash
python train.py
```
### Results on HAD100

| Bands | mAUC |
| :--: | :--: |
| The First 50 Bands | 0.9953 |
| The First 100 Bands | 0.9910 |
| The First 200 Bands | 0.9902 |


## Citation
If this repo works positively for your research, please consider citing our paper. Thanks all!
```
@article{jin2024feature,
  author={Jin, Wenping and Dang, Feng and Zhu, Li},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={Feature Enhancement with Reverse Distillation for Hyperspectral Anomaly Detection}, 
  year={2024},
  doi={10.1109/LGRS.2024.3456178}
}
```