# AGCL [TKDE 2024]

This is the implementation for the paper: “Next Point-of-Interest Recommendation with Adaptive Graph Contrastive Learning.” 

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

### Datasets

We use [SIN, NYC](https://sites.google.com/site/yangdingqi/home) and [Gowalla](http://snap.stanford.edu/data/loc-gowalla.html) datasets. The processed SIN and Gowalla datasets are from [ARGAN](https://github.com/wangzb11/AGRAN), and we preprocess NYC dataset by **data_process.py**. For more details of data preprocessing, please refer to our paper or **data_process.py**:

```
python data_process.py
```

### Graph

We use Knowledge Graph Embedding (KGE) to construct graphs. The processed code are from [Graph-Flashback](https://github.com/kevin-xuan/Graph-Flashback). For more details of graph construction, please refer to KGE directory:

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

## Citing

If you use AGCL in your research, please cite the following [paper](https://ieeexplore.ieee.org/document/10772008):
```
@article{DBLP:journals/tkde/RaoJSCHYK25,
  author       = {Xuan Rao and
                  Renhe Jiang and
                  Shuo Shang and
                  Lisi Chen and
                  Peng Han and
                  Bin Yao and
                  Panos Kalnis},
  title        = {Next Point-of-Interest Recommendation With Adaptive Graph Contrastive
                  Learning},
  journal      = {{IEEE} Trans. Knowl. Data Eng.},
  pages        = {1366--1379},
  year         = {2025}
}
```