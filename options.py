import argparse

def get_options():

    parser = argparse.ArgumentParser()
    parser.add_argument('--bandit_alg', default='exp3',help='choice of task selection algorithm:'
                                                            'exp3: Exp3,'
                                                            'exp3r:,'
                                                            'Thompson:'
                                                            'DiscountedThompson:'
                                                               )

    parser.add_argument('--warm_start', type=int, default=1, help='number of epochs to warm start ')



    parser.add_argument('--select_freq', type=int, default=None, help='Selection frequency of bandit algorithm, if None, '
                                                                      'do the selection per num_task batches, else, do '
                                                                      'the selection per select_freq*num_task batches')


    # problem setting
    # seen tasks
    parser.add_argument('--tsp', nargs='+', type=int, default=None)
    parser.add_argument('--cvrp', nargs='+', type=int, default=None)
    parser.add_argument('--op', nargs='+', type=int, default=None)
    parser.add_argument('--kp', nargs='+', type=int, default=None)
    # unseen tasks
    parser.add_argument('--unseen_tsp', nargs='+', type=int, default=None)
    parser.add_argument('--unseen_cvrp', nargs='+', type=int, default=None)
    parser.add_argument('--unseen_op', nargs='+', type=int, default=None)
    parser.add_argument('--unseen_kp', nargs='+', type=int, default=None)

    # training params
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--encoder_layer_num', type=int, default=6)

    parser.add_argument('--train_episodes', type=int, default=100*1000)
    parser.add_argument('--train_batch_size', type=int, default=64)

    parser.add_argument('--evaluation_size', type=int, default=1024)
    parser.add_argument('--model_save_interval', type=int, default=50)
    parser.add_argument('--model_load', action='store_true')
    parser.add_argument('--resume_path', type=str, default=None)
    parser.add_argument('--resume_epoch', type=int, default=None)

    parser.add_argument('--task_description', type=str, default=None)

    opts = parser.parse_args()
    return opts