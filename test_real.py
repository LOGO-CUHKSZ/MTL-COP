import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import logging
from utils import create_logger, copy_all_src
from Tester_real import COPTester as Tester
import os
import argparse


def main(opts):
    import yaml
    with open('./config.yaml') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    model_params = config['model_params']
    model_params['sqrt_embedding_dim'] = model_params['embedding_dim']**(.5)
    if opts.tsp: # read tsplib
        ds = 'tsp'
    if opts.cvrp: # read cvrplib
        ds = 'cvrp'

    tester_params = {
        'use_cuda': True if opts.device_num is not None else False,
        'cuda_device_num': opts.device_num,
        'model_load': {
            'path': opts.model_path,  # directory path of pre-trained model and log files saved.
            'epoch': opts.model_epoch,  # epoch version of pre-trained model to laod.
        },
        'test_episodes': opts.test_episodes,
        'test_batch_size': 1,
        'augmentation_enable': True if opts.aug_factor is not None else False,
        'aug_factor': opts.aug_factor,
        'aug_batch_size': 1,
    }
    if tester_params['augmentation_enable']:
        tester_params['test_batch_size'] = tester_params['aug_batch_size']

    logger_params = {
        'log_file': {
            'desc': 'test-{}-{}'.format(ds+'lib' ,opts.task_description),
            'filename': 'run_log',
            'filepath':'./result/' + '{desc}'
        }
    }

    create_logger(**logger_params)
    _print_config()

    tester = Tester(ds,model_params=model_params,
                    tester_params=tester_params,)

    copy_all_src(tester.result_folder)

    tester.run()


def _print_config():
    logger = logging.getLogger('root')
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # problem setting
    parser.add_argument('--tsp', action='store_true')
    parser.add_argument('--cvrp', action='store_true')

    parser.add_argument('--test_episodes', type=int, default=1)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--aug_factor', type=int, default=8)
    parser.add_argument('--aug_batch_size', type=int, default=1)

    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--model_epoch', type=int, default=None)
    parser.add_argument('--device_num', type=int, default=0)
    parser.add_argument('--task_description', type=str, default=None)

    opts = parser.parse_args()
    # opts.cvrp = True
    assert opts.tsp or opts.cvrp, 'At least one problem should be selected.'
    if opts.task_description is None:
        opts.task_description = opts.model_path.split('desc-')[1]

    main(opts)