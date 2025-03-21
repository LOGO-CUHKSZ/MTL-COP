import time
import sys
import os
from datetime import datetime
import logging
import logging.config
import pytz
import numpy as np
import matplotlib.pyplot as plt
import json
import shutil
import torch
import networkx as nx

import tsplib95
import pandas as pd

process_start_time = datetime.now(pytz.timezone("Asia/Shanghai"))
result_folder = './result/' + process_start_time.strftime("%Y%m%d_%H%M%S") + '{desc}'


def get_result_folder():
    return result_folder


def set_result_folder(folder):
    global result_folder
    result_folder = folder


def create_logger(log_file=None):
    if 'filepath' not in log_file:
        log_file['filepath'] = get_result_folder()

    if 'desc' in log_file:
        log_file['filepath'] = log_file['filepath'].format(desc='_' + log_file['desc'])
    else:
        log_file['filepath'] = log_file['filepath'].format(desc='')

    set_result_folder(log_file['filepath'])

    if 'filename' in log_file:
        filename = log_file['filepath'] + '/' + log_file['filename']
    else:
        filename = log_file['filepath'] + '/' + 'log.txt'

    if not os.path.exists(log_file['filepath']):
        os.makedirs(log_file['filepath'])

    file_mode = 'a' if os.path.isfile(filename)  else 'w'

    root_logger = logging.getLogger()
    root_logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(filename)s(%(lineno)d) : %(message)s", "%Y-%m-%d %H:%M:%S")

    for hdlr in root_logger.handlers[:]:
        root_logger.removeHandler(hdlr)

    # write to file
    fileout = logging.FileHandler(filename, mode=file_mode)
    fileout.setLevel(logging.INFO)
    fileout.setFormatter(formatter)
    root_logger.addHandler(fileout)

    # write to console
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    root_logger.addHandler(console)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += (val * n)
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count else 0


class LogData:
    def __init__(self):
        self.keys = set()
        self.data = {}

    def get_raw_data(self):
        return self.keys, self.data

    def set_raw_data(self, r_data):
        self.keys, self.data = r_data

    def append_all(self, key, *args):
        if len(args) == 1:
            value = [list(range(len(args[0]))), args[0]]
        elif len(args) == 2:
            value = [args[0], args[1]]
        else:
            raise ValueError('Unsupported value type')

        if key in self.keys:
            self.data[key].extend(value)
        else:
            self.data[key] = np.stack(value, axis=1).tolist()
            self.keys.add(key)

    def append(self, key, *args):
        if len(args) == 1:
            args = args[0]

            if isinstance(args, int) or isinstance(args, float):
                if self.has_key(key):
                    value = [len(self.data[key]), args]
                else:
                    value = [0, args]
            elif type(args) == tuple:
                value = list(args)
            elif type(args) == list:
                value = args
            else:
                raise ValueError('Unsupported value type')
        elif len(args) == 2:
            value = [args[0], args[1]]
        else:
            raise ValueError('Unsupported value type')

        if key in self.keys:
            self.data[key].append(value)
        else:
            self.data[key] = [value]
            self.keys.add(key)

    def get_last(self, key):
        if not self.has_key(key):
            return None
        return self.data[key][-1]

    def has_key(self, key):
        return key in self.keys

    def get(self, key):
        split = np.hsplit(np.array(self.data[key]), 2)

        return split[1].squeeze().tolist()

    def getXY(self, key, start_idx=0):
        split = np.hsplit(np.array(self.data[key]), 2)

        xs = split[0].squeeze().tolist()
        ys = split[1].squeeze().tolist()

        if type(xs) is not list:
            return xs, ys

        if start_idx == 0:
            return xs, ys
        elif start_idx in xs:
            idx = xs.index(start_idx)
            return xs[idx:], ys[idx:]
        else:
            raise KeyError('no start_idx value in X axis data.')

    def get_keys(self):
        return self.keys


class TimeEstimator:
    def __init__(self):
        self.logger = logging.getLogger('TimeEstimator')
        self.start_time = time.time()
        self.count_zero = 0

    def reset(self, count=1):
        self.start_time = time.time()
        self.count_zero = count-1

    def get_est(self, count, total):
        curr_time = time.time()
        elapsed_time = curr_time - self.start_time
        remain = total-count
        remain_time = elapsed_time * remain / (count - self.count_zero)

        elapsed_time /= 3600.0
        remain_time /= 3600.0

        return elapsed_time, remain_time

    def get_est_string(self, count, total):
        elapsed_time, remain_time = self.get_est(count, total)

        elapsed_time_str = "{:.2f}h".format(elapsed_time) if elapsed_time > 1.0 else "{:.2f}m".format(elapsed_time*60)
        remain_time_str = "{:.2f}h".format(remain_time) if remain_time > 1.0 else "{:.2f}m".format(remain_time*60)

        return elapsed_time_str, remain_time_str

    def print_est_time(self, count, total):
        elapsed_time_str, remain_time_str = self.get_est_string(count, total)

        self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
            count, total, elapsed_time_str, remain_time_str))


def util_print_log_array(logger, result_log: LogData):
    assert type(result_log) == LogData, 'use LogData Class for result_log.'

    for key in result_log.get_keys():
        logger.info('{} = {}'.format(key+'_list', result_log.get(key)))


def util_save_log_image_with_label(result_file_prefix,
                                   img_params,
                                   result_log: LogData,
                                   labels=None):
    dirname = os.path.dirname(result_file_prefix)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    _build_log_image_plt(img_params, result_log, labels)

    if labels is None:
        labels = result_log.get_keys()
    file_name = '_'.join(labels)
    fig = plt.gcf()
    fig.savefig('{}-{}.jpg'.format(result_file_prefix, file_name))
    plt.close(fig)


def _build_log_image_plt(img_params,
                         result_log: LogData,
                         labels=None):
    assert type(result_log) == LogData, 'use LogData Class for result_log.'

    # Read json
    folder_name = img_params['json_foldername']
    file_name = img_params['filename']
    log_image_config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), folder_name, file_name)

    with open(log_image_config_file, 'r') as f:
        config = json.load(f)

    figsize = (config['figsize']['x'], config['figsize']['y'])
    plt.figure(figsize=figsize)

    if labels is None:
        labels = result_log.get_keys()
    for label in labels:
        plt.plot(*result_log.getXY(label), label=label)

    ylim_min = config['ylim']['min']
    ylim_max = config['ylim']['max']
    if ylim_min is None:
        ylim_min = plt.gca().dataLim.ymin
    if ylim_max is None:
        ylim_max = plt.gca().dataLim.ymax
    plt.ylim(ylim_min, ylim_max)

    xlim_min = config['xlim']['min']
    xlim_max = config['xlim']['max']
    if xlim_min is None:
        xlim_min = plt.gca().dataLim.xmin
    if xlim_max is None:
        xlim_max = plt.gca().dataLim.xmax
    plt.xlim(xlim_min, xlim_max)

    plt.rc('legend', **{'fontsize': 18})
    plt.legend()
    plt.grid(config["grid"])


def copy_all_src(dst_root):
    # execution dir
    if os.path.basename(sys.argv[0]).startswith('ipykernel_launcher'):
        execution_path = os.getcwd()
    else:
        execution_path = os.path.dirname(sys.argv[0])

    # home dir setting
    tmp_dir1 = os.path.abspath(os.path.join(execution_path, sys.path[0]))
    tmp_dir2 = os.path.abspath(os.path.join(execution_path, sys.path[1]))

    if len(tmp_dir1) > len(tmp_dir2) and os.path.exists(tmp_dir2):
        home_dir = tmp_dir2
    else:
        home_dir = tmp_dir1

    # make target directory
    dst_path = os.path.join(dst_root, 'src')

    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    for item in sys.modules.items():
        key, value = item

        if hasattr(value, '__file__') and value.__file__:
            src_abspath = os.path.abspath(value.__file__)

            if os.path.commonprefix([home_dir, src_abspath]) == home_dir:
                dst_filepath = os.path.join(dst_path, os.path.basename(src_abspath))

                if os.path.exists(dst_filepath):
                    split = list(os.path.splitext(dst_filepath))
                    split.insert(1, '({})')
                    filepath = ''.join(split)
                    post_index = 0

                    while os.path.exists(filepath.format(post_index)):
                        post_index += 1

                    dst_filepath = filepath.format(post_index)

                shutil.copy(src_abspath, dst_filepath)


def read_score_loss(file_dir):

    info = torch.load(file_dir)

    trend = np.concatenate(info['trend_list'],axis=0)
    choices = np.array(info['choices']).reshape(-1,1)
    rewards = np.array(info['rewards']).reshape(-1,1)

    seen_tasks = file_dir.split('/')[-2].split('_')[2].split('-')
    unseen_tasks = file_dir.split('/')[-2].split('_')[4].split('-')[1:]

    res_dic = {}

    data = info['result_log'][1]
    score = np.concatenate([_[1].reshape(-1, 1) for _ in data['train_score']], axis=-1)
    loss = [_[1] for _ in data['train_loss']]

    count = 0
    for info in seen_tasks:
        temp_info = info.split('[')
        problem = temp_info[0]
        res_dic[problem] = {}
        res_dic[problem]['seen'] = {}
        scales = temp_info[1].strip(']').split(',')
        for scale in scales:
            res_dic[problem]['seen'][scale] = score[count, :]
            count += 1

    for info in unseen_tasks:
        temp_info = info.split('[')
        problem = temp_info[0]
        res_dic[problem]['unseen'] = {}
        scales = temp_info[1].strip(']').split(',')
        for scale in scales:
            res_dic[problem]['unseen'][scale] = score[count, :]
            count += 1
    assert count == score.shape[0]
    problem_list = []
    for problem in res_dic.keys():
        problem_list.append([])
        for scale in res_dic[problem]['seen'].keys():
            problem_list[-1].append((problem + '-' + scale).replace(' ',''))
        for scale in res_dic[problem]['unseen'].keys():
            problem_list[-1].append((problem + '-' + scale).replace(' ',''))
    return res_dic, loss, trend, choices, rewards, problem_list



def plot_score(data_dic, log_flag=True):
    rows = len(data_dic.keys())
    col_leng = []
    for problem in data_dic.keys():
        seen_unseen_data = data_dic[problem]
        l = 0
        if 'seen' in seen_unseen_data.keys():
            l+=len(seen_unseen_data['seen'])
        if 'unseen' in seen_unseen_data.keys():
            l+=len(seen_unseen_data['unseen'])
        col_leng.append(l)
    cols = max(col_leng)

    plt.figure(figsize=(8 * rows, 5 * cols))
    count = 1
    for problem, datass in data_dic.items():
        temp_count = 1
        for task_type, datas in datass.items():
            for scale,data in datas.items():
                if log_flag:
                    if problem == 'TSP' or problem == 'CVRP':
                        norm_val = np.min([1e6, np.min(data)])
                        data = data - norm_val + 1e-3
                    else:
                        norm_val = np.max([0, np.max(data)])
                        data = -data + norm_val + 1e-3
                    data = np.log(data)
                ax = plt.subplot(rows,cols,count)
                ax.set_title('{}-{}-{}'.format(problem,scale,task_type),fontsize=15, fontweight='bold')

                roll_data = np.roll(data,1)
                judge = ((data<roll_data)[1:])
                # ax.plot(np.arange(len(judge)),(np.cumsum(judge)/np.arange(len(judge))+1), label='approximated p for {}-{}-{}'.format(problem,scale, task_type))
                ax.plot(data, label='approximated p for {}-{}-{}'.format(problem,scale, task_type))

                plt.legend(fontsize=10)
                count+=1
                temp_count+=1
        count = count+cols+1-temp_count
    plt.tight_layout()
    plt.show()


def plot_retuen(choices, rewards, problem_list):
    unique_choice = np.unique(choices)
    rows = len(problem_list)
    cols = max([len(_) for _ in problem_list])
    plt.figure(figsize=(8 * rows, 5 * cols))
    problems = []
    for problem in problem_list:
        problems += problem
    for i, choice in enumerate(unique_choice):
        ax = plt.subplot(rows, cols, i + 1)
        idx = choices==choice
        reward = rewards[idx]
        problem = problems[i]
        ax.plot(np.cumsum(reward))
        ax.set_title('Return for Training {}'.format(problem), fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_trend(choices, trends, problem_list):
    unique_choice = np.unique(choices)
    rows = len(problem_list)
    cols = max([len(_) for _ in problem_list])
    plt.figure(figsize=(8 * rows, 5 * cols))
    problems = []
    for problem in problem_list:
        problems += problem
    for i, choice in enumerate(unique_choice):
        ax = plt.subplot(rows, cols, i + 1)
        idx = choices==choice
        trend = trends[idx.reshape(-1),:]
        problem = problems[i]
        for j in range(trend.shape[1]):
            tr = trend[:,j]
            ax.plot(np.cumsum(tr)/np.cumsum(1+np.arange(len(tr))),label='Training {} for {}'.format(problems[i],problems[j]))
        ax.set_title('Trend for Training {}'.format(problem), fontsize=15, fontweight='bold')
        plt.legend()
    plt.show()


def plt_rew(file_dir):
    info = torch.load(file_dir)
    eval_res = np.concatenate(info['eval_res'],axis=0)[:,:12]
    tasks = ['tsp20','tsp50','tsp100','cvrp20','cvrp50','cvrp100','op20','op50','op100','kp50','kp100','kp200','tsp200','tsp500','cvrp200','cvrp500','op200','op500','kp500','kp1000']
    reward = np.array(info['rewards'])
    choices = np.array(info['choices'])
    unique_choice = np.unique(choices)
    linestyles = ['-','-','-','-.','-.','-.',':',':',':','--','--','--']
    plt.figure(figsize=(8*5,5*5))
    for choice in unique_choice:
        # rew = judge.copy()
        # rew[choices!=choice] = 0
        plt.subplot(4,3,choice+1)
        rew = eval_res[choices==choice]
        for r in range(rew.shape[1]):
            plt.plot(np.cumsum(rew[:,r]), linestyle=linestyles[r],label='arm {} for task {}'.format(tasks[choice],tasks[r]))
            plt.legend()
    plt.show()


def anayl_weights_mat(file_dir, choices, trends, problem_list):

    eval_res = np.concatenate(torch.load(file_dir)['eval_res'],axis=0)

    problems = []
    for problem in problem_list:
        problems += problem

    choices = choices.reshape(-1)
    unique_choice = np.unique(choices)
    W_list = []
    for i, problem in enumerate(problems):
        res = eval_res[:,i]
        res = trends[:,i]
        W = np.zeros((len(unique_choice),len(unique_choice)))
        for j in unique_choice:
            idx_j = np.where(choices==j)[0]
            res_j = res[idx_j]
            if problem.split('-')[0] == 'KP' or problem.split('-')[0] == 'OP':
                judge_j = np.mean(res_j>res[idx_j-1])
            else:
                judge_j = np.mean(res_j<res[idx_j-1])
            for k in unique_choice:
                idx_k = np.where(choices==k)[0]
                res_k = res[idx_k]
                if problem.split('-')[0] == 'KP' or problem.split('-')[0] == 'OP':
                    judge_k = np.mean(res_k>res[idx_k-1])
                else:
                    judge_k = np.mean(res_k<res[idx_k-1])
                W[j,k] = judge_j/judge_k

        W_list.append(W)
    W_mat = np.stack(W_list,axis=0)
    prin_vec_list = []
    CR_list = []
    for W in W_list:
        eigen_val, eigen_vec = np.linalg.eig(W)
        idx = np.argmax(eigen_val)
        prin_vec = np.real(eigen_vec[:,idx])
        CR = (np.real(eigen_val[idx])-8)/7/1.41
        prin_vec_list.append(prin_vec/np.sum(prin_vec))
        CR_list.append(CR)
    prin_vec_mat = np.stack(prin_vec_list,axis=0)


    take_mat = W_mat[range(W_mat.shape[1]),range(W_mat.shape[1]),:] - 1
    test_mat = (take_mat<0).sum(1)

    pass




def read_improvement_graph(file_dir):

    info = torch.load(file_dir)
    eval_res = np.concatenate(info['trend_list'],axis=0)
    choices = info['choices']

    seen_node_idx = np.unique(choices)
    graph_size = len(eval_res[0])

    seen_task_infos = file_dir.split('/')[-2].split('_')[2].split('-')
    unseen_task_infos = file_dir.split('/')[-2].split('_')[4].split('-')[1:]

    seen_tasks = []
    total_tasks = []
    for info in seen_task_infos:
        temp_info = info.split('[')
        problem = temp_info[0]
        scales = temp_info[1].strip(']').replace(' ','').split(',')
        for scale in scales:
            seen_tasks.append(problem+'-'+scale)
            total_tasks.append(problem+'-'+scale)

    for info in unseen_task_infos:
        temp_info = info.split('[')
        problem = temp_info[0]
        scales = temp_info[1].strip(']').split(',')
        for scale in scales:
            total_tasks.append(problem+'-'+scale)

    weights = []
    unique_choice = np.unique(choices)
    plt.figure()
    for choice in unique_choice:
        idx = np.array(choices)==choice
        cum_sum = np.cumsum(idx)
        plt.plot(cum_sum,label='Train {}'.format(total_tasks[choice]))

        eval_res_choice = eval_res[idx]
        weights.append(eval_res_choice.sum(axis=0).reshape(1,-1))
    plt.legend()
    plt.show()
    plt.figure()

    weight_mat = np.concatenate(weights,axis=0)

    G = nx.DiGraph()
    edge_list = []
    for i in range(len(seen_node_idx)):
        for j in range(graph_size):
            edge_list.append((seen_tasks[i],total_tasks[j],weight_mat[i,j]))
    G.add_weighted_edges_from(edge_list)
    return G


def plot_graph(G):
    labels = nx.get_edge_attributes(G, 'weight')
    pos = nx.circular_layout(G)
    nx.draw_spring(G,with_labels=True)
    plt.show()


def read_vrp(filename):
    probelm = tsplib95.load(filename).as_name_dict()
    dimension = probelm['dimension']
    coords = np.array([probelm['node_coords'][i + 1] for i in range(probelm['dimension'])])
    capacity = probelm['capacity']
    # every element in demand is devided by capacity
    demand = np.array([probelm['demands'][i + 1] / capacity for i in range(probelm['dimension'])])
    # demand = np.array([probelm['demands'][i + 1] for i in range(probelm['dimension'])]).tolist()

    xc = coords[:, 0]
    yc = coords[:, 1]
    depot = coords[0]

    gt = 0.0
    sol_file = filename[:-4] + '.sol'
    if os.path.exists(sol_file):
        with open(sol_file, 'r') as f:
            while True:
                line = f.readline()
                if line == '':
                    break
                if "Cost" in line:
                    gt = float(line.split()[-1])
                    break

    # gt = calc_vrp_cost(depot, loc, tour)
    cities = pd.DataFrame(
        np.array(coords),
        columns=['y', 'x'],
    )[['x', 'y']]
    norm_factor = max(cities.x.max() - cities.x.min(), cities.y.max() - cities.y.min())
    norm_cities = cities.apply(lambda c: (c - c.min()) / norm_factor)[['x', 'y']].values
    #combine norm_cities and demand
    norm_cities = np.concatenate([norm_cities, np.array([demand]).T], axis=1)

    return torch.from_numpy(norm_cities).to(torch.float32).unsqueeze(0), norm_factor, gt


def load_cvrp(dir):
    real_ds_list = []
    # dir = './cvrplib/read_E'
    for file in os.listdir(dir):
        if file.split('.')[-1] == 'vrp':
            real_ds_list.append(os.path.join(dir, file))
    dataset, norm_factor, gt = zip(*[read_vrp(file) for file in real_ds_list])
    return real_ds_list, dataset, norm_factor, gt