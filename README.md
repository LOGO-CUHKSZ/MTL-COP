# MTL Baseline
 This is the official repository for reproducing the **MTL baseline** in
```
Wang C, Yu T. Efficient training of multi-task neural solver with multi-armed bandits[J]. arXiv preprint arXiv:2305.06361, 2023.
```


## Installation
### Basic environment settings
Our codes run perfectly on Ubuntu18.04 and CUDA version 11.7. The basic environment settings are as follows:
``` Bash
Python=3.10
PyTorch=2.0.1
```


# How To Run
There are various mtl_method available:
```
naive
uw
cagrad
pcgrad
nashmtl
imtl
banditmtl
```


## Training
```python
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --alg $mtl_method --tsp 20 50 100  --cvrp 20 50 100  --op 20 50 100  --kp 50 100 200 --epochs 100 --task_description mtl_baseline --model_save_interval 5
```


## Evaluation
Test on Generated Data:
```python
CUDA_VISIBLE_DEVICES=0 python test.py --model_path "your/model/path" --model_epoch 1000
```
Test on TSPLib or CVRPLib:
```python
CUDA_VISIBLE_DEVICES=0 python test_real.py --tsp --model_path "your/model/path" --model_epoch 1000
CUDA_VISIBLE_DEVICES=0 python test_real.py --cvrp --model_path "your/model/path" --model_epoch 1000
```

## Acknowledge
This code is based on the POMO repository: [POMO](https://github.com/yd-kwon/POMO)