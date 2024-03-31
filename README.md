# AGCL

This is the implementation for the paper: “Adaptive Graph Contrastive Learning for Next Point-of-Interest Recommendation.” 

## Preliminaries

### Conda Environment

```bash
  conda env create -f environment.yml
  ```

### Requirements
* Python 3.9.17 
* pytorch 1.10.0 
* pandas 2.0.3 
* numpy 1.25.1
* setuptools 59.5.0 -> pip install setuptools==59.5.0
* torch-summary 1.4.5 -> pip install torch-summary 
```bash
  pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
  ```

## Datasets

We use [SIN, NYC](https://sites.google.com/site/yangdingqi/home) and [Gowalla](http://snap.stanford.edu/data/loc-gowalla.html) datasets. The processed SIN and Gowalla datasets are from [ARGAN](https://github.com/wangzb11/AGRAN), and we preprocess NYC dataset by **data_process.py**. For more details of data preprocessing, please refer to our paper or **data_process.py**:

```
python data_process.py
```




## Model Training

To train our model with default hyper-parameters:

```
python main.py
```


## Acknowledgement

The code is implemented based on [ARGAN](https://github.com/wangzb11/AGRAN).
