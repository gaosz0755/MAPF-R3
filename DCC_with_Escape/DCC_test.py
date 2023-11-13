'''create test set and test model'''
import os
import random
import time
import argparse
import pickle
import sys
from typing import Tuple, Union
import warnings
warnings.simplefilter("ignore", UserWarning)
# from tqdm import tqdm
import numpy as np
import torch
import torch.multiprocessing as mp
from environment import Environment
from model import Network
import config

torch.manual_seed(config.test_seed)
np.random.seed(config.test_seed)
random.seed(config.test_seed)
DEVICE = torch.device('cpu')
torch.set_num_threads(1)


def test_model(model_range: Union[int, tuple], map_type,map_size, num_agents, test_num, DHM_policy, escape_policy):
    '''
    test model in 'saved_models' folder
    '''
    network = Network()
    network.eval()
    network.to(DEVICE)

    pool = mp.Pool(mp.cpu_count()//2)

    if isinstance(model_range, int):
        state_dict = torch.load('./saved_models/{}.pth'.format(model_range), map_location=DEVICE)
        network.load_state_dict(state_dict)
        network.eval()
        network.share_memory()

        
        print('----------MAPF: {} test model: {}----------'.format('DCC', model_range))
        print('the map is: {}'.format(map_type))

        if map_type == 'random_map':
            test_instance_directory = '../instances_oneshot/{}/density_0.3/instance_of_map_{}'.format(map_type, map_size)
            result_dir = '../results/one_shot/{}_0.3/RDE_DCC_DHM_{}_escape_{}'.format(map_type, DHM_policy, escape_policy)
        else:
            test_instance_directory = '../instances_oneshot/{}/instance_of_map_{}'.format(map_type, map_size)
            result_dir = '../results/one_shot/{}/RDE_DCC_DHM_{}_escape_{}'.format(map_type, DHM_policy, escape_policy)
        if not os.path.exists(result_dir):
            os.system(r"mkdir {}".format(result_dir))

        test_file = test_instance_directory + '/test{}_{}_{}.pkl'.format(map_size, num_agents, test_num)
        with open(test_file, 'rb') as f:
            tests = pickle.load(f)

        success = 0
        min_SOC = 1000000
        aveg_SOC = 0
        min_MS = 1000000
        aveg_MS = 0
        aveg_time = 0
        makespan_of_DCC = []
        SOC_of_DCC = []
        reached_goals_functions = []
        running_time = []
        index_success = []
        sum_turn_cnt = 0

        for i in range(test_num):
            terminator = 0
            reached_goals_function = []
            reached_agents = []
            reached_goals = 0
            sum_of_cost = 0
            turn_cnt = 0
            start_time = time.time()
            env = Environment(escape_policy)
            env.load(tests['maps'][i], tests['agents'][i], tests['goals'][i])
            obs, last_act, pos = env.observe()
            done = [False for _ in range(env.num_agents)]
            pre_action = [0 for _ in range(env.num_agents)]
            network.reset()
            step = 0
            num_comm = 0
            path_len = [0 for _ in range(env.num_agents)]
            while (False in done) and env.steps < config.max_episode_length:
                actions, _, _, _, comm_mask = network.step(
                    torch.as_tensor(obs.astype(np.float32)).to(DEVICE),
                    torch.as_tensor(last_act.astype(np.float32)).to(DEVICE),
                    torch.as_tensor(pos.astype(int)))
                # --------------DHM-----------------------#
                if DHM_policy:
                    for index in range(env.num_agents):
                        if done[index] == False:
                            if np.all(obs[index, 0] == False):
                                closer_action = []
                                if obs[index, 2, 4, 4] == True:
                                    closer_action.append(0)
                                if obs[index, 3, 4, 4] == True:
                                    closer_action.append(1)
                                if obs[index, 4, 4, 4] == True:
                                    closer_action.append(2)
                                if obs[index, 5, 4, 4] == True:
                                    closer_action.append(3)
                                if closer_action:
                                    # actions[index] = random.choice(closer_action)
                                    if pre_action[index] in closer_action:
                                        actions[index] = pre_action[index]
                                    else:
                                        actions[index] = random.choice(closer_action)

                (obs, last_act, pos), _, done, _ = env.step(actions)
                step += 1

                for j in range(env.num_agents):
                    if done[j] and j not in reached_agents:
                        reached_agents.append(j)
                        reached_goals += 1
                        path_len[j] = step
                    pre_action[j] = actions[j]
                reached_goals_function.append(reached_goals)

                num_comm += np.sum(comm_mask)
                for j in range(env.num_agents):
                    if done[j] and j not in reached_agents:
                        reached_agents.append(j)
                        reached_goals += 1
                        path_len[j] = step
                reached_goals_function.append(reached_goals)

            agents_path = [[] for _ in range(env.num_agents)]
            finish_id = [0 for _ in range(env.num_agents)]
            for episode in range(env.steps + 1):
                for id in range(env.num_agents):
                    if not finish_id[id]:
                        agents_path[id].append(env.history[episode][id].tolist())
                        if np.array_equal(env.history[episode][id], env.goals_pos[id]):
                            finish_id[id] = 1

            for agent_num in range(env.num_agents):
                sum_of_cost += len(agents_path[agent_num]) - 1

                n = len(agents_path[agent_num])
                if n > 2:  # 转弯
                    for step in range(2, n):
                        delta_x1 = agents_path[agent_num][step][0] - agents_path[agent_num][step - 1][0]
                        delta_y1 = agents_path[agent_num][step][1] - agents_path[agent_num][step - 1][1]
                        delta_x2 = agents_path[agent_num][step - 1][0] - agents_path[agent_num][step - 2][0]
                        delta_y2 = agents_path[agent_num][step - 1][1] - agents_path[agent_num][step - 2][1]
                        if (delta_x1 == delta_x2) and (delta_y1 == delta_y2) and not (
                                agents_path[agent_num][step] == agents_path[agent_num][step - 1]):
                            turn_cnt += 1

            sum_turn_cnt += turn_cnt

            # the timestep of reached goals function is 150
            if False in done:
                episode_time = time.time() - start_time
                makespan = max(path_len)
                sum_of_cost = sum(path_len)
                index_success.append(0)
            else:
                episode_time = time.time() - start_time
                makespan = env.steps
                for k in range(env.steps, config.max_episode_length):
                    reached_goals_function.append(reached_goals)
                sum_of_cost = sum(path_len)
                success += 1
                index_success.append(1)
            reached_goals_functions.append(reached_goals_function)

            running_time.append(episode_time)
            makespan_of_DCC.append(makespan)
            aveg_time += episode_time
            aveg_MS += makespan
            if makespan < min_MS and makespan != 0:
                min_MS = makespan
            SOC_of_DCC.append(sum_of_cost)
            aveg_SOC += sum_of_cost
            if sum_of_cost < min_SOC and sum_of_cost != 0:
                min_SOC = sum_of_cost

            if i != 0 and i % 100 == 0:
                print(" Until ",i," the success radio is ",success/(i+1)*100)
                if success / (i + 1) * 100 < 0.2:
                    terminator = 1
                    break

            # ---------------------------------------------------#
            sys.stdout.write(
                "\r map size:{} agent num:{}  test num: {}".format(map_size, num_agents, i + 1))
            sys.stdout.flush()
            # ---------------------------------------------------#
        if terminator == 1:
            aveg_time = 0
            aveg_MS = 0
            aveg_SOC = 0
            success_ratio = 0
            straight_ratio = 0
        else:
            if success != 0:
                straight_ratio = sum_turn_cnt / aveg_SOC
                aveg_time = aveg_time / test_num
                aveg_MS = aveg_MS / test_num
                aveg_SOC = aveg_SOC / test_num
                success_ratio = success / test_num * 100
            else:
                straight_ratio = 0
                aveg_time = 0
                aveg_MS = 0
                aveg_SOC = 0
                success_ratio = 0

        print(' min_SOC: {} aveg_SOC: {:.2f} success_ratio: {:.1f}% aveg_time: {:.2f} straight_ratio: {:.2f}%'.format(min_SOC,
                                                                                            aveg_SOC,
                                                                                            success_ratio,
                                                                                            aveg_time, straight_ratio * 100))

        # put the rsult into the file
        # # makespan
        with open(result_dir + '/MS_map_{}_agent_{}_testnum_{}.pk'.format(map_size, num_agents, test_num), 'wb') as f:
            pickle.dump(makespan_of_DCC, f)
        makespan_file = open(result_dir + "/MS_in_map_{}_agent_{}_testnum_{}.txt".format(map_size, num_agents, test_num), 'a')
        makespan_file.write(str(makespan_of_DCC) + '\n')
        makespan_file.close()
        # SOC
        with open(result_dir + '/SOC_map_{}_agent_{}_testnum_{}.pk'.format(map_size, num_agents, test_num), 'wb') as f:
            pickle.dump(SOC_of_DCC, f)
        SOC_file = open(result_dir + "/SOC_map_{}_agent_{}_testnum_{}.txt".format(map_size, num_agents, test_num), 'a')
        SOC_file.write(str(SOC_of_DCC) + '\n')
        SOC_file.close()
        #---------------------------------------------------------------------------------------------------------------#
        with open(result_dir + '/goals_map_{}_agent_{}_testnum_{}.pk'.format(map_size, num_agents, test_num), 'wb') as f:
            pickle.dump(reached_goals_functions, f)
        reach_goals_file = open(
            result_dir + "/goals_map_{}_agent_{}_testnum_{}.txt".format(map_size, num_agents, test_num), 'a')
        reach_goals_file.write(str(reached_goals_functions) + '\n')
        reach_goals_file.close()
        #--------------------------------------------------------------------------------------------------------------#
        with open(result_dir + '/index_success_map_{}_agent_{}_testnum_{}.pk'.format(
                map_size, num_agents, test_num),
                  'wb') as f:
            pickle.dump(index_success, f)
        file = open(
            result_dir + "/index_success_map_{}_agent_{}_testnum_{}.txt".format(
                map_size, num_agents, test_num), 'a')
        file.write(str(index_success) + '\n')
        file.close()
        #---------------------------------------------------------------------------------------------------------------#
        result_file = open(result_dir + "/result_{}_{}_map_agent_{}_testnum_{}.txt".format(
            map_type, map_size, num_agents, test_num), 'a')
        result_file.write('min_SOC:' + str(min_SOC) + '\n' + 'aveg_SOC:' + str(aveg_SOC) + '\n' + 'min_MS:' + str(
            min_MS) + '\n' + 'aveg_MS:' + str(aveg_MS) + '\n' + 'success_ratio:' + str(
            success_ratio) + '\n' + 'aveg_time:' + str(aveg_time) + '\n' + 'straight_ratio:'+str(straight_ratio))
        result_file.close()

        return success_ratio

if __name__ == '__main__':


    test_num = 1000

    map_types = ['narrow_aisle_warehouse_2_2'] # , 'wide_aisle_warehouse', 'free_map','narrow_aisle_warehouse_2_2'

    DHM_policy = False
    escape_policy = False
    print('DHM_policy:', DHM_policy, ' escape_policy:', escape_policy)

    for map_type in map_types:
        # print('map type is:', map_type)
        for map_size in [34]:  #  34, 100, 200
            if map_size < 100:
                test_num = 1000
            if map_size >= 100:
                test_num = 50
            # 10,20,30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150
            for num_agents in [10,30,50,70]:  # 8,16,32,64  10,30,50,70,
                # load trained model and reproduce results in paper
                success_ratio = test_model(128000, map_type,map_size, num_agents, test_num, DHM_policy, escape_policy)
                if success_ratio == 0:
                    break
