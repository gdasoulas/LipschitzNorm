#!/usr/bin/python

"""
Launch a grid search batch of experiments base on config_xps.json
"""

import os
import pathlib
import json
import time
import argparse
import itertools
from utils import safe_mkdir
import shutil

def launch(_args, _args_value, _outputdir, start, xp, _tot_xp):
    cmd = ' python ./train_deepgat_depth.py --' + ' --'.join([str(a) + ' ' + str(v) for a, v in zip(_args, _args_value)])
    cmd += ' --outputdir {} --xp {}'.format(_outputdir, xp)
    print('##########################################################################')
    print('### STARTED {} ### '.format(start) + time.strftime("(%Y/%m/%d at %H:%M:%S)") +
          ' Launching ({}/{}): '.format(xp + 1, _tot_xp) + cmd)
    os.system(cmd)


def get_best_cv_result(xps_path):
    xps = next(os.walk(xps_path))[1]
    results = [json.load(open(os.path.join(xps_path, xp, 'logs.json')))['cv_test_acc_mean'] for xp in xps]

    best_res = max(results)
    best_res_index = results.index(best_res)
    best_xp = xps[best_res_index]
    return best_res, best_xp


def main(xps):
    for ds in xps['dataset']:
        # -- create result directory
        now = time.strftime("%Y%m%d_%H%M%S")
        outputdir = os.path.join(os.getcwd(), 'experiments',ds, ds + '_' + now)
        safe_mkdir(outputdir)

        # -- save grid
        json.dump(xps, open(os.path.join(outputdir, 'config_launched_xps.json'), 'w'),
                  indent=4, sort_keys=False)


        # -- copy executed python files
        pathlib.Path(os.path.join(outputdir,'executed_files')).mkdir(parents=True, exist_ok=True)
        shutil.copy2('./train_cora.py', os.path.join(outputdir, 'executed_files/train_cora.py') )
        shutil.copy2('./utils.py', os.path.join(outputdir, 'executed_files/utils.py') )
        shutil.copytree('./models', os.path.join(outputdir, 'executed_files/models/') )

        # -- get list of xps arguments
        args = []
        values = []
        for i, (key, value) in enumerate(xps.items()):
            args.append(key)
            if key == 'dataset':
                values.append([ds])
            else:
                values.append(value)

            if key == 'nepoch':
                epoch_ind = i
            if key == 'num_layers':
                layer_ind = i

        combination_list = list(itertools.product(*values))
        tot_xp = len(combination_list)

        # -- launch xps
        print('######################LAUNCHING BATCH OF EXPERIMENTS######################')
        curr_xp_id = 0
        for args_value in combination_list:
            if (args_value[layer_ind] <= 4 and args_value[epoch_ind] != 1000 ):
                continue

            if (args_value[layer_ind] > 4 and args_value[layer_ind] < 8 and args_value[epoch_ind] != 1500 ):
                continue

            if (args_value[layer_ind] >= 8 and args_value[epoch_ind] != 2500 ):
                continue

            launch(args, args_value, outputdir, now, curr_xp_id, tot_xp)
            curr_xp_id += 1

        # -- get best xp based on cv
        # best_res, best_xp = get_best_cv_result(outputdir)
        # best_res_str = '### Best cross validated result: {} with xp {}'.format(best_res, best_xp)
        # with open(os.path.join(outputdir, 'best_cv_res.txt'), 'w') as f:
        #     f.write(best_res_str)
        # print(best_res_str)


if __name__ == '__main__':
    # -- parse mandatory argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_xps', default=None, type=str,
                        help='global config json file path for a batch of experiments (default: None)')
    args_p = parser.parse_args()

    assert args_p.config_xps is not None, "Please give a json file with experiments to launch"
    xps_to_launch = json.load(open(args_p.config_xps))
    main(xps_to_launch)

