import pickle
import numpy as np



def compare_with_bm_same_budget():
    def print_latex(budget='small'):
        uni_dir = './result/_test-TSP[20, 50, 100]-CVRP[20, 50, 100]-OP[20, 50, 100]-KP[50, 100, 200]-train12task_exp3_freq12'
        mtl_dir = './result/_test-TSP[20, 50, 100]-CVRP[20, 50, 100]-OP[20, 50, 100]-KP[50, 100, 200]-MTL_baseline'

        banditmtl_dir = './result/_test-TSP[20, 50, 100]-CVRP[20, 50, 100]-OP[20, 50, 100]-KP[50, 100, 200]-bs_banditmtl'
        nashmtl_dir = './result/_test-TSP[20, 50, 100]-CVRP[20, 50, 100]-OP[20, 50, 100]-KP[50, 100, 200]-bs_nashmtl'
        pcgrad_dir = './result/_test-TSP[20, 50, 100]-CVRP[20, 50, 100]-OP[20, 50, 100]-KP[50, 100, 200]-bs_pcgrad'
        uw_dir = './result/_test-TSP[20, 50, 100]-CVRP[20, 50, 100]-OP[20, 50, 100]-KP[50, 100, 200]-bs_uw'
        cagrad_dir = './result/_test-TSP[20, 50, 100]-CVRP[20, 50, 100]-OP[20, 50, 100]-KP[50, 100, 200]-bs_cagrad'
        imtl_dir = './result/_test-TSP[20, 50, 100]-CVRP[20, 50, 100]-OP[20, 50, 100]-KP[50, 100, 200]-bs_imtl'


        if budget == 'small':
            with open(uni_dir + '/result_gap_epoch{}-best.pkl'.format(
                        500), 'rb') as f:
                data = pickle.load(f)

            epoch_w = epoch_small_weigted
            epoch_e = epoch_small_equal
            with open(mtl_dir + '/result_gap_epoch{}-best.pkl'.format(
                        50), 'rb') as f:
                mtl_data = pickle.load(f)
            with open(banditmtl_dir + '/result_gap_epoch{}-best.pkl'.format(
                        36), 'rb') as f:
                banditmtl_data = pickle.load(f)
            with open(nashmtl_dir + '/result_gap_epoch{}-best.pkl'.format(
                        6), 'rb') as f:
                nashmtl_data = pickle.load(f)
            with open(pcgrad_dir + '/result_gap_epoch{}-best.pkl'.format(
                        6), 'rb') as f:
                pcgrad_data = pickle.load(f)
            with open(uw_dir + '/result_gap_epoch{}-best.pkl'.format(
                        37), 'rb') as f:
                uw_data = pickle.load(f)
            with open(cagrad_dir + '/result_gap_epoch{}-best.pkl'.format(
                        7), 'rb') as f:
                cagrad_data = pickle.load(f)
            with open(imtl_dir + '/result_gap_epoch{}-best.pkl'.format(
                        6), 'rb') as f:
                imtl_data = pickle.load(f)

        elif budget == 'median':
            with open(
                    uni_dir + '/result_gap_epoch{}-best.pkl'.format(
                        1000), 'rb') as f:
                data = pickle.load(f)
                epoch_w = epoch_median_weighted
                epoch_e = epoch_median_equal
            with open(mtl_dir + '/result_gap_epoch{}-best.pkl'.format(
                        100), 'rb') as f:
                mtl_data = pickle.load(f)
            with open(banditmtl_dir + '/result_gap_epoch{}-best.pkl'.format(
                        75), 'rb') as f:
                banditmtl_data = pickle.load(f)
            with open(nashmtl_dir + '/result_gap_epoch{}-best.pkl'.format(
                        13), 'rb') as f:
                nashmtl_data = pickle.load(f)
            with open(pcgrad_dir + '/result_gap_epoch{}-best.pkl'.format(
                        13), 'rb') as f:
                pcgrad_data = pickle.load(f)
            with open(uw_dir + '/result_gap_epoch{}-best.pkl'.format(
                        76), 'rb') as f:
                uw_data = pickle.load(f)
            with open(cagrad_dir + '/result_gap_epoch{}-best.pkl'.format(
                        14), 'rb') as f:
                cagrad_data = pickle.load(f)
            with open(imtl_dir + '/result_gap_epoch{}-best.pkl'.format(
                        14), 'rb') as f:
                imtl_data = pickle.load(f)
        elif budget == 'large':
            with open(
                    uni_dir
                    + '/result_gap_epoch{}-best.pkl'.format(
                        2000), 'rb') as f:
                data = pickle.load(f)
                epoch_w = epoch_large_weigted
                epoch_e = epoch_large_eqaul
            with open(mtl_dir + '/result_gap_epoch{}-best.pkl'.format(
                        200), 'rb') as f:
                mtl_data = pickle.load(f)
            with open(banditmtl_dir + '/result_gap_epoch{}-best.pkl'.format(
                        153), 'rb') as f:
                banditmtl_data = pickle.load(f)
            with open(nashmtl_dir + '/result_gap_epoch{}-best.pkl'.format(
                        27), 'rb') as f:
                nashmtl_data = pickle.load(f)
            with open(pcgrad_dir + '/result_gap_epoch{}-best.pkl'.format(
                        26), 'rb') as f:
                pcgrad_data = pickle.load(f)
            with open(uw_dir + '/result_gap_epoch{}-best.pkl'.format(
                        155), 'rb') as f:
                uw_data = pickle.load(f)
            with open(cagrad_dir + '/result_gap_epoch{}-best.pkl'.format(
                        29), 'rb') as f:
                cagrad_data = pickle.load(f)
            with open(imtl_dir + '/result_gap_epoch{}-best.pkl'.format(
                        28), 'rb') as f:
                imtl_data = pickle.load(f)

        uni_aug_res = data['aug_gap']

        mtl_aug_res = mtl_data['aug_gap']
        banditmtl_aug_res = banditmtl_data['aug_gap']
        nashmtl_aug_res = nashmtl_data['aug_gap']
        pcgrad_aug_res = pcgrad_data['aug_gap']
        uw_aug_res = uw_data['aug_gap']
        cagrad_aug_res = cagrad_data['aug_gap']
        imtl_aug_res = imtl_data['aug_gap']


        idx = [0,1,2] * 4

        no_aug_res_w = []
        aug_res_w = []
        file_epoch = zip((epoch_w.flatten().tolist())*(len(newpaths)//12), newpaths)
        for epoch, res_file in file_epoch:
            no_aug_res_w.append([])
            aug_res_w.append([])
            with open(res_file + '/result_gap_epoch{}.pkl'.format(epoch), 'rb') as f:
                data = pickle.load(f)
            no_aug_res_w[-1].append(data['no_aug_gap'])
            aug_res_w[-1].append(data['aug_gap'])
            no_aug_res_w[-1] = np.concatenate(no_aug_res_w[-1], axis=0)
            aug_res_w[-1] = np.concatenate(aug_res_w[-1], axis=0)

        aug_res_w = np.stack(aug_res_w, axis=0).reshape(-1,12,3).mean(0)[range(12),idx]


        no_aug_res_e = []
        aug_res_e = []
        file_epoch = zip((epoch_e.flatten().tolist())*(len(newpaths)//12), newpaths)
        for epoch, res_file in file_epoch:
            no_aug_res_e.append([])
            aug_res_e.append([])
            with open(res_file + '/result_gap_epoch{}.pkl'.format(epoch), 'rb') as f:
                data = pickle.load(f)
            no_aug_res_e[-1].append(data['no_aug_gap'])
            aug_res_e[-1].append(data['aug_gap'])
            no_aug_res_e[-1] = np.concatenate(no_aug_res_e[-1], axis=0)
            aug_res_e[-1] = np.concatenate(aug_res_e[-1], axis=0)

        aug_res_e = np.stack(aug_res_e, axis=0).reshape(-1,12,3).mean(0)[range(12),idx]

        # ratio_w = uni_aug_res - aug_res_w
        # ratio_e = uni_aug_res - aug_res_e
        #
        # mtl_ratio_w = mtl_aug_res - aug_res_w
        # mtl_ratio_e = mtl_aug_res - aug_res_e
        #
        # banditmtl_ratio_w = banditmtl_aug_res - aug_res_w
        # banditmtl_ratio_e = banditmtl_aug_res - aug_res_e
        #
        # nashmtl_ratio_w = nashmtl_aug_res - aug_res_w
        # nashmtl_ratio_e = nashmtl_aug_res - aug_res_e
        #
        # pcgrad_ratio_w = pcgrad_aug_res - aug_res_w
        # pcgrad_ratio_e = pcgrad_aug_res - aug_res_e
        #
        # uw_ratio_w = uw_aug_res - aug_res_w
        # uw_ratio_e = uw_aug_res - aug_res_e
        return aug_res_e, aug_res_w, mtl_aug_res, banditmtl_aug_res, pcgrad_aug_res, uw_aug_res, cagrad_aug_res, imtl_aug_res, nashmtl_aug_res, uni_aug_res,
        # \
        #     mtl_ratio_e, mtl_ratio_w, banditmtl_ratio_e, banditmtl_ratio_w, pcgrad_ratio_e, pcgrad_ratio_w, uw_ratio_e, \
        #     uw_ratio_w, nashmtl_ratio_e, nashmtl_ratio_w, ratio_e, ratio_w
        # return ratio_e, ratio_w, uni_aug_res, aug_res_e, aug_res_w, mtl_aug_res, mtl_ratio_e, mtl_ratio_w

    res_files = [
        './result/_test-TSP[20, 50, 100]-bm_tsp20',
        './result/_test-TSP[20, 50, 100]-bm_tsp50',
        './result/_test-TSP[20, 50, 100]-bm_tsp100',
        './result/_test-CVRP[20, 50, 100]-bm_cvrp20',
        './result/_test-CVRP[20, 50, 100]-bm_cvrp50',
        './result/_test-CVRP[20, 50, 100]-bm_cvrp100',
        './result/_test-OP[20, 50, 100]-bm_op20',
        './result/_test-OP[20, 50, 100]-bm_op50',
        './result/_test-OP[20, 50, 100]-bm_op100',
        './result/_test-KP[50, 100, 200]-bm_kp50',
        './result/_test-KP[50, 100, 200]-bm_kp100',
        './result/_test-KP[50, 100, 200]-bm_kp200',
    ]


    newpaths = []
    for repeat in ['_repeat{}_save'.format(i+1) for i in range(3)]:
        for file in res_files:
            newpaths.append(file+repeat)

    problem_list = ['TSP-20', 'TSP-50', 'TSP-100', 'CVRP-20', 'CVRP-50', 'CVRP-100', 'OP-20', 'OP-50', 'OP-100',
                    'KP-50', 'KP-100', 'KP-200']

    epoch_small_weigted = np.array([[61, 61, 47],
                          [44, 48, 39],
                         [59, 58, 59],
                          [35, 39, 32],])

    epoch_small_equal = np.array([
                                    [123, 61, 31],
                                    [89, 48, 26],
                                    [118, 58, 39],
                                    [70, 39, 21]])

    epoch_median_weighted = np.array([[127, 125, 98],
                                     [92, 99, 82],
                                      [122, 120, 122],
                                     [72, 81, 67]])

    epoch_median_equal = np.array([[254, 125, 65],
                                     [185, 99, 54],
                                    [244, 120, 81],
                                     [144, 81, 45]])


    epoch_large_weigted = np.array([[256, 254, 199],
                                   [187, 200, 165],
                                    [247, 243, 246],
                                   [146, 163, 136]])

    epoch_large_eqaul = np.array([[513, 254, 133],
                                   [374, 200, 110],
                                  [494, 243, 164],
                                   [292, 163,  90]])



    res_s = print_latex('small')
    aug_res_e_s, aug_res_w_s, mtl_aug_res_s, banditmtl_aug_res_s, pcgrad_aug_res_s, uw_aug_res_s, cagrad_aug_res_s, imtl_aug_res_s, nashmtl_aug_res_s, uni_aug_res_s = res_s
    res_m = print_latex('median')
    aug_res_e_m, aug_res_w_m, mtl_aug_res_m, banditmtl_aug_res_m, pcgrad_aug_res_m, uw_aug_res_m, cagrad_aug_res_m, imtl_aug_res_m, nashmtl_aug_res_m, uni_aug_res_m = res_m
    res_l = print_latex('large')
    aug_res_e_l, aug_res_w_l, mtl_aug_res_l, banditmtl_aug_res_l, pcgrad_aug_res_l, uw_aug_res_l, cagrad_aug_res_l, imtl_aug_res_l, nashmtl_aug_res_l, uni_aug_res_l = res_l

    res_gap = np.stack([aug_res_e_s,aug_res_w_s,mtl_aug_res_s, banditmtl_aug_res_s, pcgrad_aug_res_s, uw_aug_res_s, cagrad_aug_res_s, imtl_aug_res_s, nashmtl_aug_res_s, uni_aug_res_s,
                    aug_res_e_m,aug_res_w_m, mtl_aug_res_m, banditmtl_aug_res_m, pcgrad_aug_res_m, uw_aug_res_m,  cagrad_aug_res_m, imtl_aug_res_m, nashmtl_aug_res_m, uni_aug_res_m,
                    aug_res_e_l,aug_res_w_l, mtl_aug_res_l, banditmtl_aug_res_l, pcgrad_aug_res_l, uw_aug_res_l, cagrad_aug_res_l, imtl_aug_res_l, nashmtl_aug_res_l,uni_aug_res_l],0)
    
    total_gap =res_gap.mean(1,keepdims=True)
    
    total_res = np.concatenate([res_gap,total_gap],1)
    res_gap_l = np.split(total_res,3,0)
    best_idx_l = [np.argmin(res_gap_l[i], 0) for i in range(3)]
    

    method_list = ['$\\text{STL}_{\\text{avg.}}$','$\\text{STL}_{\\text{bal.}}$','MTL','Bandit-MTL', 'PCGrad', 'UW',
                   'CAGrad','IMTL',
                   'Nash-MTL','Ours'] * 3

    for i in range(30):
        if i %10==0:
            print('\midrule')
            print('\midrule')
            if i //10==0:
                print('\multirow{9}{*}{\\rotatebox[origin = c]{90}{Small Budget}}')
                best_idx = best_idx_l[0]
            elif i//10==1:
                print('\multirow{9}{*}{\\rotatebox[origin = c]{90}{Median Budget}}')
                best_idx = best_idx_l[1]
            else:
                print('\multirow{9}{*}{\\rotatebox[origin = c]{90}{Large Budget}}')
                best_idx = best_idx_l[2]

        print('& {}'.format(method_list[i]), end=' ')
        
        res = total_res[i]
        
        for j, r in enumerate(res):
            if i%10==best_idx[j]:
                print('& $\mathbf{{{:.3f}}}\%$ '.format(r), end=' ')
            else:
                print('& ${:.3f}\%$ '.format(r), end=' ')
        print('\\\\')


    # print('\midrule')
    # print('\midrule')
    # print('& Total Gap', end=' ')
    # for i in range(3):
    #     idx_gap = np.argmin(res_gap[i*4:(i+1)*4])
    #     for _ in range(4):
    #         if _==idx_gap:
    #             print('& $\mathbf{{{:.3f}}}\%$ '.format(res_gap[_+i*4]), end=' ')
    #         else:
    #             print('& ${:.3f}\%$ '.format(res_gap[_+i*4]), end=' ')
    # print('\\\\')
    # print('\midrule')
    # print('& Gain by MTL & ${{{:.3f}}}\%$ & ${:.3f}\%$  & - & - & ${{{:.3f}}}\%$ & ${:.3f}\%$  & - & - & ${{{:.3f}}}\%$ & ${:.3f}\%$  & - & -  \\\\'.format(*mtl_res_diff))
    # print('& Gain by Ours & $\mathbf{{{:.3f}}}\%$ & $\mathbf{{{:.3f}}}\%$  & - & - & $\mathbf{{{:.3f}}}\%$ & $\mathbf{{{:.3f}}}\%$  & -& - & $\mathbf{{{:.3f}}}\%$ & $\mathbf{{{:.3f}}}\%$  & - & -  \\\\'.format(*res_diff))
    # print('\midrule')
    # print('\midrule')


if __name__ == '__main__':
    compare_with_bm_same_budget()
