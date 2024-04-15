# Efficient Training of Multi-task Neural Solver with Multi-armed Bandits



## Installation
### Basic environment settings
Our code run perfectly on Ubuntu18.04, CUDA version 11.4 and python3.8. The basic environment settings are as follows:
``` Bash
Python=3.8
PyTorch=1.10.2
```

# How To Run

## run 

## Training
```python
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --epochs 1000 --warm_start 1 --select_freq 12 --tsp 20 50 100  --cvrp 20 50 100  --op 20 50 100  --kp 50 100 200 --bandit_alg exp3 --task_description train12task_exp3_freq12
```
## Evaluation
```python
CUDA_VISIBLE_DEVICES=0 python test.py --model_path ./result/'_train_TSP[20, 50, 100]-CVRP[20, 50, 100]-OP[20, 50, 100]-KP[50, 100, 200]_BanditAlg-exp3_unseen-_desc-train12task_exp3_freq12' --model_epoch 1000
```
