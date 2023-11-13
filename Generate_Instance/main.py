
# generate instances for oneshot MAPF

import numpy as np
import torch
import pickle
import os
import sys
import time
import random

from environment import Environment
import config

def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_test_instance(type_map, map_size, num_agents, test_num):

    if type_map == 'random_map':
        test_instance_directory = '../instances_oneshot/{}/density_{}/instance_of_map_{}'.format(type_map, config.obstacle_density, map_size)
    else:
        test_instance_directory = '../instances_oneshot/{}/instance_of_map_{}'.format(type_map, map_size)
    if not os.path.exists(test_instance_directory):
        os.system(r"mkdir {}".format(test_instance_directory))
    # 测试算例文件
    test_instance_file = test_instance_directory + '/test{}_{}_{}.pkl'.format(map_size, num_agents, test_num)
    tests = {'maps': [], 'agents': [], 'goals': []}

    env = Environment(type_map, map_size, num_agents)
    # calculate the time of D3QN distmatrix 
    # init_get_dist_time = time.time()
    # env.get_dist_map()
    # get_dist_time = time.time() - init_get_dist_time

    for num in range(test_num):
        tests['maps'].append(np.copy(env.map))
        tests['agents'].append(np.copy(env.agents_pos))
        tests['goals'].append(np.copy(env.goals_pos))
        #---------------------------------------------------#
        sys.stdout.write("\r the test number is: %d" % (num + 1))
        sys.stdout.flush()
        #---------------------------------------------------#
        env.reset(map_size)
    
    # output test instances
    with open(test_instance_file, 'wb') as f:
        pickle.dump(tests, f)

if __name__ == "__main__":

    seed_torch(1)
    # ['wide_aisle_warehouse', 'narrow_aisle_warehouse', 'free_map', 'random_map']
    types_map = ['narrow_aisle_warehouse_2_2', 'narrow_aisle_warehouse_3_2']
    # 'wide_aisle_warehouse','narrow_aisle_warehouse_2_2', 'narrow_aisle_warehouse_3_2','free_map',

    for type_map in types_map:
        print('map type is:', type_map)
        # for map_size in range (34, 71, 12): [34, 58, 82]
        for map_size in [100]:  # 34, 58, 82,
            if map_size < 100:
                test_num = 1000
            else:
                test_num = 100
            for num_agents in [1000]:  # 10,30,50,70,90  ,700,900, 1100
                if map_size < 100 and num_agents > 100:
                    continue
                if map_size >= 100 and num_agents < 100:
                    continue
                if type_map == 'random_map':
                    allow_obs_density = 1-num_agents*2/(map_size*map_size)
                    # print('mapsize is:', map_size, 'agent num is:',num_agents,'obstacle density is:', config.obstacle_density)
                    if config.obstacle_density > allow_obs_density:
                        raise RuntimeError('obstacle density is too large!!')
                generate_test_instance(type_map, map_size, num_agents, test_num)
                print('\n')