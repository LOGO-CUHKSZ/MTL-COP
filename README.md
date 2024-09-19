# Efficient Training of Multi-task Neural Solver for Combinatorial Optimization
 This is the official repository for
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
Then you need to setup `SMPyBandits` to support the bandit algorithm: Go to `./MTL-COP`, then 
```Bash
git clone https://github.com/SMPyBandits/SMPyBandits.git
```


# How To Run

## Dataset
You can download the dataset from [here](https://drive.google.com/file/d/151UOyIU7hLamwoEnXquNjc6heNm3-WuS/view?usp=sharing).


## Training
```python
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --epochs 1000 --warm_start 1 --select_freq 12 --tsp 20 50 100  --cvrp 20 50 100  --op 20 50 100  --kp 50 100 200 --bandit_alg exp3 --task_description train12task_exp3_freq12
```

## Resume the training
```python
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --epochs 1000 --warm_start 1 --select_freq 12 --tsp 20 50 100  --cvrp 20 50 100  --op 20 50 100  --kp 50 100 200 --bandit_alg exp3 --task_description train12task_exp3_freq12_resume --model_load --resume_path "your/resume/path" --resume_epoch 1000
```

## Evaluation
```python
CUDA_VISIBLE_DEVICES=0 python test.py --model_path "your/model/path" --model_epoch 1000
```
Test on TSPLib or CVRPLib:
```python
CUDA_VISIBLE_DEVICES=0 python test_real.py --tsp --model_path "your/model/path" --model_epoch 1000
CUDA_VISIBLE_DEVICES=0 python test_real.py --cvrp --model_path "your/model/path" --model_epoch 1000
```
