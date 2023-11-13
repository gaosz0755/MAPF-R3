'''create test set and test model'''
import random
import os
import pickle
import time
import sys
import multiprocessing as mp
from typing import Union
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from environment import Environment
from model import Network
import configs

torch.manual_seed(configs.test_seed)
np.random.seed(configs.test_seed)
random.seed(configs.test_seed)
test_num = 200
device = torch.device('cpu')
torch.set_num_threads(1)


def test_model(model_range: Union[int, tuple]):
    '''
    test model in 'models' file with model number 
    '''
    network = Network()
    network.eval()
    network.to(device)

    test_set = configs.test_env_settings

    pool = mp.Pool(mp.cpu_count())

    if isinstance(model_range, int):
        state_dict = torch.load('./models/{}.pth'.format(model_range), map_location=device)
        network.load_state_dict(state_dict)
        network.eval()
        network.share_memory()

        print('----------MAPF: {} test model: {}----------'.format('DHC', model_range))
        # env_types = ['narrow_aisle_warehouse', 'wide_aisle_warehouse', 'random_map']
        env_types = ['narrow_aisle_warehouse_2_2','narrow_aisle_warehouse_3_2'] # 'wide_aisle_warehouse' 'random_map' 'free_map','narrow_aisle_warehouse_2_2',

        DHM_policy = True
        escape_policy = True
        print('DHM_policy:',DHM_policy,' escape_policy:',escape_policy)

        obstacle_density_random = 0.3
        for env_type in env_types:
            print('the env type is: {}'.format(env_type))

            for map_size in [34]: # 34, 82 ,100,200
                if map_size < 100:
                    test_num = 1000
                if map_size >=100:
                    test_num = 100

                for num_agent in [10, 30, 50, 70]: # 10, 30, 50, 70, 90, 300,500,700,900,1100
                    if env_type == 'random_map':
                        tests_dir = '../instances_oneshot/{}/density_{}/instance_of_map_{}'.format(env_type, obstacle_density_random, map_size)
                        result_dir = '../results/one_shot/{}_{}/RDE_DHC_DHM_{}_escape_{}'.format(env_type, obstacle_density_random, DHM_policy, escape_policy)
                    else:
                        tests_dir = '../instances_oneshot/{}/instance_of_map_{}'.format(env_type, map_size)
                        result_dir = '../results/one_shot/{}/RDE_DHC_DHM_{}_escape_{}'.format(env_type, DHM_policy, escape_policy)
                    tests_file = tests_dir + '/test{}_{}_{}.pkl'.format(map_size, num_agent, test_num)
                    with open(tests_file, 'rb') as f:
                        tests = pickle.load(f)

                    if not os.path.exists(result_dir):
                        os.system(r"mkdir {}".format(result_dir))

                    success = 0
                    min_SOC = 100000
                    aveg_SOC = 0
                    min_MS = 100000
                    aveg_MS = 0
                    aveg_time = 0
                    makespan_of_DHC = []
                    SOC_of_DHC = []
                    reached_goals_functions = []
                    running_time = []
                    index_success = []
                    sum_turn_cnt = 0

                    for i in range(test_num):
                        reached_goals_function = []
                        reached_agents = []
                        reached_goals = 0
                        sum_of_cost = 0
                        turn_cnt = 0
                        start_time = time.time()

                        env = Environment(escape_policy)
                        env.load(tests['maps'][i], tests['agents'][i], tests['goals'][i])


                        if i == 0:
                            cnt = 0
                            for i in range(map_size):
                                for j in range(map_size):
                                    if env.map[i][j] != 0:
                                        cnt += 1
                            obs_density = cnt/(map_size * map_size)
                            print('the obs density is: {:.1f}%'.format(obs_density * 100))

                        obs, pos = env.observe()

                        done = [False for _ in range(env.num_agents)]
                        pre_action = [0 for _ in range(env.num_agents)]

                        network.reset()
                        step = 0
                        path_len = [0 for _ in range(env.num_agents)]
                        history_step = [[] for _ in range(env.num_agents)]
                        while (False in done) and env.steps < configs.max_episode_length:
                            actions, _, _, _ = network.step(torch.as_tensor(obs.astype(np.float32)), torch.as_tensor(pos.astype(np.float32)))
                            #--------------DHM-----------------------#
                            if DHM_policy:
                                for index in range(env.num_agents):
                                    if done[index] == False:
                                        if np.all(obs[index, 0] == False):
                                            closer_action = []
                                            if obs[index, 2, 4, 4] == True:
                                                closer_action.append(1)
                                            if obs[index, 3, 4, 4] == True:
                                                closer_action.append(2)
                                            if obs[index, 4, 4, 4] == True:
                                                closer_action.append(3)
                                            if obs[index, 5, 4, 4] == True:
                                                closer_action.append(4)
                                            if closer_action:
                                                # actions[index] = random.choice(closer_action)
                                                if pre_action[index] in closer_action:
                                                    actions[index] = pre_action[index]
                                                else:
                                                    actions[index] = random.choice(closer_action)

                            (obs, pos), _, done, _ = env.step(actions)

                            step += 1
                            for j in range(env.num_agents):
                                if done[j] and j not in reached_agents:
                                    reached_agents.append(j)
                                    reached_goals += 1
                                    path_len[j] = step
                                pre_action[j] = actions[j]
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
                            episode_time = 0
                            makespan = max(path_len)
                            sum_of_cost = sum(path_len)
                            index_success.append(0)
                        else:
                            episode_time = time.time() - start_time
                            makespan = env.steps
                            for k in range(env.steps, 150):
                                reached_goals_function.append(reached_goals)
                            sum_of_cost = sum(path_len)
                            success += 1
                            index_success.append(1)
                        reached_goals_functions.append(reached_goals_function)

                        running_time.append(episode_time)
                        makespan_of_DHC .append(makespan)
                        aveg_time += episode_time
                        aveg_MS += makespan
                        if makespan < min_MS and makespan !=0:
                            min_MS = makespan
                        SOC_of_DHC.append(sum_of_cost)
                        aveg_SOC += sum_of_cost
                        if sum_of_cost < min_SOC and sum_of_cost != 0:
                            min_SOC = sum_of_cost

                        if i != 0 and i % 100 == 0:
                            print(" Until ", i, " the success radio is ", success / (i + 1) * 100)

                        #---------------------------------------------------#
                        sys.stdout.write(
                            "\r map size:{} agent num:{} test num: {}".format(map_size, num_agent, i + 1))
                        sys.stdout.flush()
                        #---------------------------------------------------#
                    if success != 0:
                        straight_ratio = sum_turn_cnt / aveg_SOC
                        aveg_time = aveg_time / test_num
                        aveg_MS = aveg_MS / test_num
                        aveg_SOC = aveg_SOC / test_num

                    else:
                        straight_ratio = 0
                        aveg_time = 0
                        aveg_MS = 0
                        aveg_SOC = 0

                    print(' min_SOC: {} aveg_SOC: {:.2f} success_ratio: {:.1f}% aveg_time: {:.2f} straight_ratio: {:.2f}%'.format(min_SOC,
                                                                                                          aveg_SOC,
                                                                                                          success / test_num * 100,
                                                                                                          aveg_time, straight_ratio * 100))

                    # makespan
                    with open(result_dir + '/MS_map_{}_agent_{}_testnum_{}.pk'.format(map_size, num_agent, test_num), 'wb') as f:
                        pickle.dump(makespan_of_DHC, f)
                    makespan_file = open(result_dir + "/MS_map_{}_agent_{}_testnum_{}.txt".format(map_size, num_agent, test_num), 'a')
                    makespan_file.write(str(makespan_of_DHC) + '\n')
                    makespan_file.close()
                    #
                    # SOC
                    with open(result_dir + '/SOC_map_{}_agent_{}_testnum_{}.pk'.format(map_size, num_agent, test_num), 'wb') as f:
                        pickle.dump(SOC_of_DHC, f)
                    SOC_file = open(result_dir + "/SOC_map_{}_agent_{}_testnum_{}.txt".format(map_size, num_agent, test_num), 'a')
                    SOC_file.write(str(SOC_of_DHC) + '\n')
                    SOC_file.close()
                    #
                    # # time
                    # with open(result_dir + '/time_map_{}_agent_{}_testnum_{}.pk'.format(map_size, num_agent, test_num), 'wb') as f:
                    #     pickle.dump(running_time, f)
                    # time_file = open(result_dir + "/time_map_{}_agent_{}_testnum_{}.txt".format(map_size, num_agent, test_num), 'a')
                    # time_file.write(str(running_time) + '\n')
                    # time_file.close()
                    #
                    with open(result_dir + '/goals_map_{}_agent_{}_testnum_{}.pk'.format(map_size, num_agent, test_num), 'wb') as f:
                        pickle.dump(reached_goals_functions, f)
                    reach_goals_file = open(result_dir + "/goals_map_{}_agent_{}_testnum_{}.txt".format(map_size, num_agent, test_num), 'a')
                    reach_goals_file.write(str(reached_goals_functions) + '\n')
                    reach_goals_file.close()

                    with open(result_dir + '/index_success_map_{}_agent_{}_testnum_{}.pk'.format(
                                                                                      map_size, num_agent, test_num),
                              'wb') as f:
                        pickle.dump(index_success, f)
                    file = open(
                        result_dir + "/index_success_map_{}_agent_{}_testnum_{}.txt".format(
                                                                                 map_size, num_agent, test_num), 'a')
                    file.write(str(index_success) + '\n')
                    file.close()

                    #
                    result_file = open(result_dir + "/result_map_{}_agent_{}_testnum_{}.txt".format(map_size, num_agent, test_num), 'a')
                    result_file.write('min_SOC:' + str(min_SOC) + '\n' + 'aveg_SOC:' + str(aveg_SOC) + '\n' + 'min_MS:' + str(
                        min_MS) + '\n' + 'aveg_MS:' + str(aveg_MS) + '\n' + 'success_ratio:' + str(
                        success/test_num) + '\n' + 'aveg_time:' + str(aveg_time) + '\n' + 'straight_ratio:'+str(straight_ratio))
                    result_file.close()



    elif isinstance(model_range, tuple):
        for model_name in range(model_range[0], model_range[1]+1, configs.save_interval):
            state_dict = torch.load('./models/{}.pth'.format(model_name), map_location=device)
            network.load_state_dict(state_dict)
            network.eval()
            network.share_memory()

            print('----------test model {}----------'.format(model_name))

            for case in test_set:
                print("test set: {} length {} agents {} density".format(case[0], case[1], case[2]))
                with open('./test_set/{}length_{}agents_{}density.pth'.format(case[0], case[1], case[2]), 'rb') as f:
                    tests = pickle.load(f)

                tests = [(test, network) for test in tests]
                ret = pool.map(test_one_case, tests)

                success = 0
                avg_step = 0
                for i, j in ret:
                    success += i
                    avg_step += j

                print("success rate: {:.2f}%".format(success/len(ret)*100))
                print("average step: {}".format(avg_step/len(ret)))
                print()

            print('\n')


def test_one_case(args):

    env_set, network = args

    env = Environment()
    env.load(env_set[0], env_set[1], env_set[2])
    obs, pos = env.observe()
    
    done = False
    network.reset()

    step = 0
    while not done and env.steps < configs.max_episode_length:
        actions, _, _, _ = network.step(torch.as_tensor(obs.astype(np.float32)), torch.as_tensor(pos.astype(np.float32)))
        (obs, pos), _, done, _ = env.step(actions)
        step += 1

    return np.array_equal(env.agents_pos, env.goals_pos), step

def make_animation(model_name: int, test_set_name: tuple, test_case_idx: int, steps: int = 25):
    '''
    visualize running results
    model_name: model number in 'models' file
    test_set_name: (length, num_agents, density)
    test_case_idx: int, the test case index in test set
    steps: how many steps to visualize in test case
    '''
    color_map = np.array([[255, 255, 255],   # white
                    [190, 190, 190],   # gray
                    [0, 191, 255],   # blue
                    [255, 165, 0],   # orange
                    [0, 250, 154]])  # green

    network = Network()
    network.eval()
    network.to(device)
    state_dict = torch.load('models/{}.pth'.format(model_name), map_location=device)
    network.load_state_dict(state_dict)

    test_name = 'test_set/40_length_16_agents_0.3_density.pkl'
    with open(test_name, 'rb') as f:
        tests = pickle.load(f)

    env = Environment()
    env.load(tests[test_case_idx][0], tests[test_case_idx][1], tests[test_case_idx][2])

    fig = plt.figure()
            
    done = False
    obs, pos = env.observe()

    imgs = []
    while not done and env.steps < steps:
        imgs.append([])
        map = np.copy(env.map)
        for agent_id in range(env.num_agents):
            if np.array_equal(env.agents_pos[agent_id], env.goals_pos[agent_id]):
                map[tuple(env.agents_pos[agent_id])] = 4
            else:
                map[tuple(env.agents_pos[agent_id])] = 2
                map[tuple(env.goals_pos[agent_id])] = 3
        map = map.astype(np.uint8)

        img = plt.imshow(color_map[map], animated=True)

        imgs[-1].append(img)

        for i, ((agent_x, agent_y), (goal_x, goal_y)) in enumerate(zip(env.agents_pos, env.goals_pos)):
            text = plt.text(agent_y, agent_x, i, color='black', ha='center', va='center')
            imgs[-1].append(text)
            text = plt.text(goal_y, goal_x, i, color='black', ha='center', va='center')
            imgs[-1].append(text)


        actions, _, _, _ = network.step(torch.from_numpy(obs.astype(np.float32)).to(device), torch.from_numpy(pos.astype(np.float32)).to(device))
        (obs, pos), _, done, _ = env.step(actions)
        # print(done)

    if done and env.steps < steps:
        map = np.copy(env.map)
        for agent_id in range(env.num_agents):
            if np.array_equal(env.agents_pos[agent_id], env.goals_pos[agent_id]):
                map[tuple(env.agents_pos[agent_id])] = 4
            else:
                map[tuple(env.agents_pos[agent_id])] = 2
                map[tuple(env.goals_pos[agent_id])] = 3
        map = map.astype(np.uint8)

        img = plt.imshow(color_map[map], animated=True)
        for _ in range(steps-env.steps):
            imgs.append([])
            imgs[-1].append(img)
            for i, ((agent_x, agent_y), (goal_x, goal_y)) in enumerate(zip(env.agents_pos, env.goals_pos)):
                text = plt.text(agent_y, agent_x, i, color='black', ha='center', va='center')
                imgs[-1].append(text)
                text = plt.text(goal_y, goal_x, i, color='black', ha='center', va='center')
                imgs[-1].append(text)


    ani = animation.ArtistAnimation(fig, imgs, interval=600, blit=True, repeat_delay=1000)

    ani.save('videos/{}_{}_{}_{}.mp4'.format(model_name, *test_set_name))


if __name__ == '__main__':

    # create_test(test_env_settings=configs.test_env_settings, num_test_cases=configs.num_test_cases)
    # test_model((2000, 6000))
    test_model(337500)
