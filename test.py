import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import logging
from utils import create_logger, copy_all_src
from Tester import COPTester as Tester
import os
import argparse
import numpy as np
from tqdm import tqdm
import re



def main(opts):
    import yaml
    with open('./config.yaml') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    model_params = config['model_params']
    model_params['sqrt_embedding_dim'] = model_params['embedding_dim']**(.5)
    test_env_params = {
    }
    if opts.tsp is not None:
        test_env_params['TSP'] = {}
        test_env_params['TSP']['problem_size'] = opts.tsp
        test_env_params['TSP']['pomo_size'] = opts.tsp
    if opts.cvrp is not None:
        test_env_params['CVRP'] = {}
        test_env_params['CVRP']['problem_size'] = opts.cvrp
        test_env_params['CVRP']['pomo_size'] = opts.cvrp
    if opts.op is not None:
        test_env_params['OP'] = {'prize_type': 'dist'}
        test_env_params['OP']['problem_size'] = opts.op
        test_env_params['OP']['pomo_size'] = opts.op
    if opts.kp is not None:
        test_env_params['KP'] = {}
        test_env_params['KP']['problem_size'] = opts.kp
        test_env_params['KP']['pomo_size'] = opts.kp

    problem_list = list(test_env_params.keys())

    tester_params = {
        'use_cuda': True if opts.device_num is not None else False,
        'cuda_device_num': opts.device_num,
        'model_load': {
            'path': opts.model_path,  # directory path of pre-trained model and log files saved.
            'epoch': opts.model_epoch,  # epoch version of pre-trained model to laod.
        },
        'test_episodes': opts.test_episodes,
        'test_batch_size': opts.test_batch_size,
        'augmentation_enable': True if opts.aug_factor is not None else False,
        'aug_factor': opts.aug_factor,
        'aug_batch_size': opts.aug_batch_size,
    }
    if tester_params['augmentation_enable']:
        tester_params['test_batch_size'] = tester_params['aug_batch_size']

    logger_params = {
        'log_file': {
            'desc': 'test-{}-{}'.format('-'.join(str(_)+str(test_env_params[_]['problem_size']) for _ in problem_list),opts.task_description),
            'filename': 'run_log',
            'filepath':'./result/' + '{desc}'
        }
    }

    create_logger(**logger_params)
    _print_config()

    tester = Tester(test_env_params=test_env_params,
                    model_params=model_params,
                    tester_params=tester_params,)

    copy_all_src(tester.result_folder)

    tester.run()


def _print_config():
    logger = logging.getLogger('root')
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


def get_tasks(path):
    folder_name = path.split('/')[-1]
    pattern = r'(TSP|CVRP|OP|KP)\[([^\]]+)\]'
    matches = re.findall(pattern, folder_name)
    scales = {}
    for task_type, scale_str in matches:
        scales[task_type] = [int(s.strip()) for s in scale_str.split(',')]
    return scales


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # problem setting
    parser.add_argument('--tsp', nargs='+', type=int, default=None)
    parser.add_argument('--cvrp', nargs='+', type=int, default=None)
    parser.add_argument('--op', nargs='+', type=int, default=None)
    parser.add_argument('--kp', nargs='+', type=int, default=None)

    parser.add_argument('--test_episodes', type=int, default=10000)
    parser.add_argument('--test_batch_size', type=int, default=500)
    parser.add_argument('--aug_factor', type=int, default=8)
    parser.add_argument('--aug_batch_size', type=int, default=500)

    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--model_epoch', type=int, default=None)
    parser.add_argument('--device_num', type=int, default=0)
    parser.add_argument('--task_description', type=str, default=None)

    opts = parser.parse_args()
    opts.task_description = opts.model_path.split('desc-')[1]

    seen_cop = get_tasks(opts.model_path)
    all_problem = list(set(list(seen_cop.keys())))
    for problem in all_problem:
        if problem == 'TSP':
            opts.tsp = np.sort(seen_cop[problem]).tolist()

        elif problem == 'CVRP':
            opts.cvrp = np.sort(seen_cop[problem]).tolist()

        elif problem == 'KP':
            opts.kp = np.sort(seen_cop[problem]).tolist()
        elif problem == 'OP':
            opts.op = np.sort(seen_cop[problem]).tolist()

    main(opts)