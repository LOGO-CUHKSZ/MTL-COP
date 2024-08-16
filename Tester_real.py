from logging import getLogger
from Env.COPEnv import COPEnv as Env
from Models.models import COPModel as Model
from utils import *
import pickle
import tsplib95
import pandas as pd
from utils import load_cvrp

def read_tsplib(filename):
    """
    Read a file in .tsp format into a pandas DataFrame

    The .tsp files can be found in the TSPLIB project. Currently, the library
    only considers the possibility of a 2D map.
    """
    probelm = tsplib95.load(filename).as_name_dict()
    coords = np.array([probelm['node_coords'][i+1] for i in range(probelm['dimension'])])
    optinfo = (np.array(tsplib95.load(filename[:-4]+'.opt.tour').tours)-1).reshape(-1)
    opt_seq_coords = coords[optinfo]
    gt = np.sqrt(np.square(
                opt_seq_coords - np.concatenate([opt_seq_coords[1:], opt_seq_coords[0].reshape(1, -1)], axis=0)).sum(
                -1)).sum()
    cities = pd.DataFrame(
        np.array(coords),
        columns=['y', 'x'],
    )[['x', 'y']]

    norm_factor = max(cities.x.max() - cities.x.min(), cities.y.max() - cities.y.min())
    norm_cities = cities.apply(lambda c: (c - c.min()) / norm_factor)[['x', 'y']].values
    return torch.from_numpy(norm_cities).to(torch.float32).unsqueeze(0), norm_factor, gt


class COPTester:
    def __init__(self,
                 ds,
                 model_params,
                 tester_params,
                 ):

        model_load = tester_params['model_load']

        args_file = '{path}/args.json'.format(**model_load)
        with open(args_file, 'r') as f:
            args = json.load(f)


        # load dataset
        if ds == 'tsp':
            self.real_ds_list = []
            for file in os.listdir('datasets/TSP/tsplib'):
                if file.split('.')[-1] == 'tsp':
                    instance_name = file.split('.')[0]
                    # judge whether there is an opt file:
                    if os.path.exists('datasets/TSP/tsplib/' + instance_name + '.opt.tour'):
                        self.real_ds_list.append('./datasets/TSP/tsplib/' + file)
            self.test_data, self.norm_factor, self.gt = zip(*[read_tsplib(file) for file in self.real_ds_list])
            test_env_params = {'TSP': {'problem_size': [data.shape[1] for data in self.test_data], 'pomo_size': [min([data.shape[1],100]) for data in self.test_data]}}
        else:
            self.real_ds_list = []
            self.test_data = []
            self.norm_factor = []
            self.gt = []
            for cvrp_dir in os.listdir('datasets/CVRP/cvrplib'):
                real_ds_list, test_data, norm_factor, gt = load_cvrp("datasets/CVRP/cvrplib/"+cvrp_dir)
                self.real_ds_list += real_ds_list
                self.test_data += test_data
                self.norm_factor += norm_factor
                self.gt += gt
            test_env_params = {'CVRP': {'problem_size': [data.shape[1] for data in self.test_data], 'pomo_size': [min([data.shape[1],100]) for data in self.test_data]}}            

        self.test_env_params = test_env_params
        self.problem = []
        try:
            if args['tsp'] is not None:
                self.problem.append('TSP')
            if args['cvrp'] is not None:
                self.problem.append('CVRP')

            if args['op'] is not None:
                self.problem.append('OP')

            if args['kp'] is not None:
                self.problem.append('KP')
        except:
            if 'tsp' in args:
                self.problem.append('TSP')

            if 'cvrp' in args:
                self.problem.append('CVRP')

            if 'op' in args:
                self.problem.append('OP')
            if 'kp' in args:
                self.problem.append('KP')

        self.test_problem = list(self.test_env_params.keys())

        self.model_params = model_params
        try:
            self.model_params['encoder_layer_num'] = args['encoder_layer_num']
        except:
            pass
        self.tester_params = tester_params

        # result folder, logger
        self.logger = getLogger(name='tester')
        self.result_folder = get_result_folder()
        print(self.result_folder)

        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        self.test_data = [data.to(device) for data in self.test_data]

        # ENV and MODEL
        self.env_list = Env(**self.test_env_params).env_list
        self.model = Model(self.problem,**self.model_params)

        # Restore
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)

        self.model.load_state_dict(checkpoint['model_state_dict'])

        # utility
        self.time_estimator = TimeEstimator()
        self.model_epoch = model_load['epoch']

    def run(self, best_mode=False):
        if os.path.exists('{}/result_gap_epoch{}.pkl'.format(self.result_folder, self.model_epoch)):
            print('{}/result_gap_epoch{}.pkl already exists'.format(self.result_folder, self.model_epoch))
            return

        self.time_estimator.reset()

        test_num_episode = self.tester_params['test_episodes']
        episode = 0

        while episode < test_num_episode:

            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            score_list, aug_score_list = self._test_one_batch(batch_size,episode,best_mode)
            gap_list = []
            aug_gap_list = []
            for i, cop_score in enumerate(score_list):
                problem = self.test_problem[i]
                for j, score in enumerate(cop_score):
                    aug_score =  aug_score_list[i][j]
                    gt = self.gt[j]
                    norm_factor = self.norm_factor[j]
                    if problem == 'KP' or problem == 'OP':
                        gap = (1 - score/gt).mean().item()*100
                        aug_gap = (1-aug_score/gt).mean().item()*100
                    else:
                        gap = (score*norm_factor/gt-1).mean().item()*100
                        aug_gap = (aug_score*norm_factor/gt-1).mean().item()*100
                    gap_list.append(gap)
                    aug_gap_list.append(aug_gap)

            episode += batch_size

            ############################
            # Logs
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], gap:{}%, aug_gap:{}%".format(
                episode, test_num_episode, elapsed_time_str, remain_time_str, gap_list, aug_gap_list))

            all_done = (episode == test_num_episode)
            no_aug_gap = {ds.split('/')[-1].split('.')[0]:gap_list[i] for i, ds in enumerate(self.real_ds_list) if gap_list[i]>0}
            aug_gap = {ds.split('/')[-1].split('.')[0]:aug_gap_list[i] for i, ds in enumerate(self.real_ds_list) if aug_gap_list[i]>0}

            if all_done:
                self.logger.info(" *** Test Done *** ")
                self.logger.info(" NO-AUG SCORE: {} ".format(gap_list))
                self.logger.info(" Mean NO-AUG SCORE: {} ".format(np.mean(gap_list)))
                self.logger.info(" AUGMENTATION SCORE: {} ".format(aug_gap_list))
                self.logger.info(" Mean AUGMENTATION SCORE: {} ".format(np.mean(aug_gap_list)))
                result = {'no_aug_gap':no_aug_gap,
                          'aug_gap':aug_gap}

                with open('{}/result_gap_epoch{}.pkl'.format(self.result_folder,self.model_epoch), 'wb') as file:
                    pickle.dump(result, file)




    def _test_one_batch(self, batch_size,episode,best_mode=False):

        # Augmentation
        ###############################################
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1

        # Ready
        ###############################################
        self.model.eval()
        for i,cop_env in enumerate(self.env_list[0]):
            cop_data = self.test_data[i]
            cop_env.load_problems(batch_size,aug_factor,prepare_dataset=cop_data)

        # reset_state, _, _ = zip(*[env.reset() for env in self.env_list])
        reset_state = []
        states, rewards, dones = [], [], []
        for cop_env in self.env_list:
            temp_reset_state = []
            temp_state = []
            temp_reward = []
            temp_dones = []
            for env in cop_env:
                reset_s, _, _ = env.reset()
                state, reward, done = env.pre_step()
                temp_reset_state.append(reset_s)
                temp_state.append(state)
                temp_reward.append(reward)
                temp_dones.append(done)

            reset_state.append(temp_reset_state)
            states.append(temp_state)
            rewards.append(temp_reward)
            dones.append(temp_dones)

        no_aug_score_list, aug_score_list = [], []
        with torch.no_grad():
            for k in range(len(self.env_list)):
                no_aug_score_list.append([])
                aug_score_list.append([])
                cop_env = self.env_list[k]
                problem = self.test_problem[k]

                for i in range(len(cop_env)):
                    env = cop_env[i]
                    state, reward, done = states[k][i], rewards[k][i], dones[k][i]
                    self.model.pre_forward_oneCOP(reset_state[k][i], problem)
                    while not done:
                        selected, _ = self.model(state, problem)
                        # shape: (batch, pomo)
                        state, reward, done = env.step(selected)


                    # Return
                    ###############################################
                    aug_reward = reward.reshape(aug_factor, env.batch_size//aug_factor, env.pomo_size)
                    # shape: (augmentation, batch, pomo)

                    max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
                    # shape: (augmentation, batch)
                    no_aug_score = torch.abs(max_pomo_reward[0, :].float())  # negative sign to make positive value

                    max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
                    # shape: (batch,)
                    aug_score = torch.abs(max_aug_pomo_reward.float())  # negative sign to make positive value

                    no_aug_score_list[-1].append(no_aug_score.cpu().numpy())
                    aug_score_list[-1].append(aug_score.cpu().numpy())
        return no_aug_score_list, aug_score_list

    def get_atten_weights(self):
        test_num_episode = self.tester_params['test_episodes']
        episode = 0
        atten_weights = []
        while episode < test_num_episode:
            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            atten_weights_one_batch = self.get_atten_weights_one_batch(batch_size)
            atten_weights.append(atten_weights_one_batch)
            episode += batch_size
        return torch.cat(atten_weights,dim=0)

    def get_atten_weights_one_batch(self, batch_size):
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1
        # Ready
        ###############################################
        self.model.eval()
        [env.load_problems(batch_size, aug_factor) for env in self.env_list]
        self.env_list[0].problems = self.env_list[1].depot_node_xy[:, 1:, :]
        reset_state, _, _ = zip(*[env.reset() for env in self.env_list])
        with torch.no_grad():
            atten_weights = self.model.get_atten_weights(reset_state)
        return atten_weights
