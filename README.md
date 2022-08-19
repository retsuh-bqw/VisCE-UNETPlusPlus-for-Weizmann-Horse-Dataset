# VisCE-UNETPlusPlus-for-Weizmann-Horse-Dataset

## Prepare for Data

1. Weizmann Horse dataset can be directly downloaded on [Weizmann Horse Database | Kaggle](https://www.kaggle.com/datasets/ztaihong/weizmann-horse-database?resource=download)

2. The dataset can be unpacked anywhere you like, just specify the dataset root at training time

## Inference

Run the following command to do inference of UNET++ Weizmann Horse dataset:

```shell
python test.py --weights path-to-your-weights-file
```

Our model for reporting results is available at the following link:

Link：https://pan.baidu.com/s/1Iq194ymXP-lHPYrT9d8W_g?pwd=hust 
Code：hust

The visualized results of the model is output by default in the *./demo* folder of the project root.

## Training

Run the following command to train UNET++:

```sh
python train.py --data_root dataset-root-path --out_dir log-output-dir --save_path saved-weight-path
```

Default training parameters:

Initial learning rate=1e-4, batch size=16, epoch=100, weight decay=5e-4

- The training log is output to the *./log*  in the root directory by default
- The weight files obtained from training will be output to the *./checkpoint* in the root directory by default
- You can use pre-trained weights with the option *--pretrained*

## Permission and Disclaimer

This code is only for non-commercial purposes. The trained models included in this repository can only be used/distributed for non-commercial purposes. Anyone who violates this rule will be at his/her own risk.