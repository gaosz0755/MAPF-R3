import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib import colors
from numba import jit
import config
import random
import time
import heapq
from AStar import check_map_using_Astar


from typing import List

action_list = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]],
                       dtype=np.int8)

def move(loc, dir):
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    return loc[0] + directions[dir][0], loc[1] + directions[dir][1]

def check_map(map, agents_pos, goals_pos):
        # check whether every agent can find its path from start to goal position using A*
        # temp_agents_pos = np.copy(agents_pos)
        # agents_pos= []
        # for pos in temp_agents_pos:
        #     agents_pos.append(tuple(pos))

        # temp_goals_pos = np.copy(goals_pos)
        # goals_pos = []
        # for pos in temp_goals_pos:
        #     goals_pos.append(tuple(pos))
        cnt = 0
        map_is_legal = False
        for i in range(len(agents_pos)):
            temp_pass = check_map_using_Astar(map, agents_pos[i], goals_pos[i])
            if temp_pass == True:
                cnt += 1
            else:
                break
        print(cnt)
        if cnt == len(agents_pos):
            map_is_legal = True

        return map_is_legal

def check_map_density(map):
        obs_cnt = 0
        map_size = len(map)
        for i in range(map_size):
            for j in range(map_size):
                if map[i][j] == 1:
                    obs_cnt += 1
        obs_density = obs_cnt / (map_size*map_size)
        return obs_cnt, obs_density

class Environment:
    def __init__(self, type_map, map_size, num_agents):

        self.env_type = type_map
        self.num_agents = num_agents  #  Variable(number_of_agent) is in the "test" file
        self.map_size = tuple((map_size,map_size))

        if self.env_type == 'random_map':

            map_is_legal = False
            i = 0
            while (map_is_legal == False):
                print('i:',i+1)
                G = np.zeros(self.map_size[0]*self.map_size[1])  
                obs_num = int(G.size * config.obstacle_density)
                obs_a = np.random.randint(0, high=G.size, size=obs_num)
                G[obs_a] = 1
                self.map = G.reshape([self.map_size[0], self.map_size[1]])

                count = 0
                self.agents_pos, self.goals_pos = [], []
                while (count < self.num_agents):
                    # temp_place = [start,goal]
                    temp_start = (np.random.randint(self.map_size[0]),
                                np.random.randint(self.map_size[0]))
                    temp_end = (np.random.randint(self.map_size[0]),
                                np.random.randint(self.map_size[0]))
                    if temp_start not in self.agents_pos and temp_start not in self.goals_pos and self.map[temp_start[0], temp_start[
                        1]] == 0 and temp_end not in self.agents_pos and temp_end not in self.goals_pos and self.map[temp_end[0], temp_end[1]] == 0:
                        self.agents_pos.append(temp_start)
                        self.goals_pos.append(temp_end)
                        count += 1
                self.agents_pos = np.array(self.agents_pos)
                self.goals_pos = np.array(self.goals_pos)

                map_is_legal = check_map(self.map, self.agents_pos, self.goals_pos)
                i=i+1

        if self.env_type == 'wide_aisle_warehouse' or self.env_type == 'narrow_aisle_warehouse_2_2' or self.env_type == 'narrow_aisle_warehouse_3_2':
            if self.env_type == 'wide_aisle_warehouse':
                self.map = np.zeros(self.map_size).astype(np.float32)
                for i in range(config.start_place, self.map_size[0], config.step_stride):
                    for j in range(config.start_place,self.map_size[0], config.step_stride):
                        self.map[i:i + 2, j:j + 2] = 1
            if self.env_type == 'narrow_aisle_warehouse_2_2':
                self.map = np.zeros(self.map_size).astype(np.float32)
                start_place = 1
                step_stride_row = 3
                step_stride_column = 3
                for i in range(start_place, self.map_size[0], step_stride_column):
                    for j in range(start_place, self.map_size[0], step_stride_row):
                        self.map[i:i + 2, j:j + 2] = 1
                obs_cnt, obs_density = check_map_density(self.map)
                print('obs_density is:', obs_density)
            if self.env_type == 'narrow_aisle_warehouse_3_2':
                self.map = np.zeros(self.map_size).astype(np.float32)
                start_place = 1
                step_stride_row = 3
                step_stride_column = 4
                for i in range(start_place, self.map_size[0], step_stride_column):
                    for j in range(start_place, self.map_size[0], step_stride_row):
                        self.map[i:i + 3, j:j + 2] = 1
                obs_cnt, obs_density = check_map_density(self.map)
                print('obs_density is:', obs_density)
            # choose start and goal place
            count = 0
            self.agents_pos, self.goals_pos = [], []
            while (count < self.num_agents):
                # temp_place = [start,goal]
                temp_start = (np.random.randint(self.map_size[0]),
                            np.random.randint(self.map_size[0]))
                temp_end = (np.random.randint(1, self.map_size[0]),
                            np.random.randint(1, self.map_size[0]))
                if temp_start not in self.agents_pos and self.map[
                    temp_start[0], temp_start[
                        1]] == 0 and temp_end not in self.goals_pos and self.map[
                    temp_end[0], temp_end[1]] == 1:
                    self.agents_pos.append(temp_start)
                    self.goals_pos.append(temp_end)
                    count += 1
            self.agents_pos = np.array(self.agents_pos)
            self.goals_pos = np.array(self.goals_pos)


        if self.env_type == 'free_map':
            self.map = np.zeros(self.map_size).astype(np.float32)
            generate_pos = False
            while not generate_pos:
                count = 0
                self.agents_pos, self.goals_pos = [], []
                while (count < self.num_agents):
                    # temp_place = [start,goal]
                    temp_start = (np.random.randint(self.map_size[0]),
                                np.random.randint(self.map_size[0]))
                    temp_end = (np.random.randint(self.map_size[0]),
                                np.random.randint(self.map_size[0]))
                    if temp_start not in self.agents_pos and temp_start not in self.goals_pos and temp_end not in self.agents_pos and temp_end not in self.goals_pos:
                        self.agents_pos.append(temp_start)
                        self.goals_pos.append(temp_end)
                        count += 1
                self.agents_pos = np.array(self.agents_pos)
                self.goals_pos = np.array(self.goals_pos)
                for i in range(len(self.agents_pos)):
                    if np.array_equal(self.agents_pos[i], self.goals_pos[i]):
                        generate_pos = False
                        break
                    else:
                        generate_pos = True


        # self.get_dist_map()


    def reset(self, dis):
        dis = int(dis)
        self.num_agents = self.num_agents
        if self.env_type == 'random_map':

            map_is_legal = False
            while (map_is_legal == False):
                G = np.zeros(self.map_size[0]*self.map_size[1])  
                obs_num = int(G.size * config.obstacle_density)
                obs_a = np.random.randint(0, high=G.size, size=obs_num)
                G[obs_a] = 1
                self.map = G.reshape([self.map_size[0], self.map_size[1]])

                count = 0
                self.agents_pos, self.goals_pos = [], []
                while (count < self.num_agents):
                    # temp_place = [start,goal]
                    temp_start = (np.random.randint(self.map_size[0]),
                                np.random.randint(self.map_size[0]))
                    temp_end = (np.random.randint(self.map_size[0]),
                                np.random.randint(self.map_size[0]))
                    if (temp_start not in self.agents_pos) and (temp_start not in self.goals_pos) and (self.map[temp_start[0], temp_start[
                        1]] == 0) and (temp_end not in self.agents_pos) and (temp_end not in self.goals_pos) and (self.map[temp_end[0], temp_end[1]] == 0):
                        self.agents_pos.append(temp_start)
                        self.goals_pos.append(temp_end)
                        count += 1
                self.agents_pos = np.array(self.agents_pos)
                self.goals_pos = np.array(self.goals_pos)

                map_is_legal = check_map(self.map, self.agents_pos, self.goals_pos)

        if self.env_type == 'wide_aisle_warehouse' or self.env_type == 'narrow_aisle_warehouse_2_2' or self.env_type == 'narrow_aisle_warehouse_3_2':
            if self.env_type == 'wide_aisle_warehouse':
                self.map = np.zeros(self.map_size).astype(np.float32)
                for i in range(config.start_place, self.map_size[0], config.step_stride):
                    for j in range(config.start_place,self.map_size[0], config.step_stride):
                        self.map[i:i + 2, j:j + 2] = 1
            if self.env_type == 'narrow_aisle_warehouse_2_2':
                self.map = np.zeros(self.map_size).astype(np.float32)
                start_place = 1
                step_stride_row = 3
                step_stride_column = 3
                for i in range(start_place, self.map_size[0], step_stride_column):
                    for j in range(start_place, self.map_size[0], step_stride_row):
                        self.map[i:i + 2, j:j + 2] = 1
            if self.env_type == 'narrow_aisle_warehouse_3_2':
                self.map = np.zeros(self.map_size).astype(np.float32)
                start_place = 1
                step_stride_row = 3
                step_stride_column = 4
                for i in range(start_place, self.map_size[0], step_stride_column):
                    for j in range(start_place, self.map_size[0], step_stride_row):
                        self.map[i:i + 3, j:j + 2] = 1
            # choose start and goal place
            count = 0
            self.agents_pos, self.goals_pos = [], []
            while (count < self.num_agents):
                # temp_place = [start,goal]
                temp_start = (np.random.randint(self.map_size[0]),
                            np.random.randint(self.map_size[0]))
                temp_end = (np.random.randint(1, self.map_size[0]),
                            np.random.randint(1, self.map_size[0]))
                if temp_start not in self.agents_pos and self.map[
                    temp_start[0], temp_start[
                        1]] == 0 and temp_end not in self.goals_pos and self.map[
                    temp_end[0], temp_end[1]] == 1:
                    self.agents_pos.append(temp_start)
                    self.goals_pos.append(temp_end)
                    count += 1
            self.agents_pos = np.array(self.agents_pos)
            self.goals_pos = np.array(self.goals_pos)

        if self.env_type == 'free_map':
            self.map = np.zeros(self.map_size).astype(np.float32)
            generate_pos = False
            while not generate_pos:
                count = 0
                self.agents_pos, self.goals_pos = [], []
                while (count < self.num_agents):
                    # temp_place = [start,goal]
                    temp_start = (np.random.randint(self.map_size[0]),
                                np.random.randint(self.map_size[0]))
                    temp_end = (np.random.randint(self.map_size[0]),
                                np.random.randint(self.map_size[0]))
                    if temp_start not in self.agents_pos and temp_start not in self.goals_pos and temp_end not in self.agents_pos and temp_end not in self.goals_pos:
                        self.agents_pos.append(temp_start)
                        self.goals_pos.append(temp_end)
                        count += 1
                self.agents_pos = np.array(self.agents_pos)
                self.goals_pos = np.array(self.goals_pos)
                for i in range(len(self.agents_pos)):
                    if np.array_equal(self.agents_pos[i], self.goals_pos[i]):
                        generate_pos = False
                        break
                    else:
                        generate_pos = True

    

    def get_dist_map(self):
        # Use Dijkstra to build a shortest-path tree rooted at the goal location

        self.dist_map = np.ones((self.num_agents, self.map_size[0], self.map_size[1]),dtype=np.int32) * 2147483647
        for i in range(self.num_agents):
            goal = (self.goals_pos[i,0], self.goals_pos[i,1])
            open_list = []
            closed_list = dict()
            root = {'loc': goal, 'cost': 0}
            heapq.heappush(open_list, (root['cost'], goal, root))
            closed_list[goal] = root
            while len(open_list) > 0:
                (cost, loc, curr) = heapq.heappop(open_list)
                for dir in range(4):
                    child_loc = move(loc, dir)
                    child_cost = cost + 1

                    if child_loc[0] < 0 or child_loc[0] >= self.map.shape[0] or \
                            child_loc[1] < 0 or child_loc[1] >= self.map.shape[1]:
                        continue

                    if self.map[child_loc[0], child_loc[1]] == 1:
                        continue

                    child = {'loc': child_loc, 'cost': child_cost}
                    if child_loc in closed_list:
                        existing_node = closed_list[child_loc]
                        if existing_node['cost'] > child_cost:
                            closed_list[child_loc] = child
                            # open_list.delete((existing_node['cost'], existing_node['loc'], existing_node))
                            heapq.heappush(open_list,
                                           (child_cost, child_loc, child))
                    else:
                        closed_list[child_loc] = child
                        heapq.heappush(open_list, (child_cost, child_loc, child))

            # build the heuristics table
            # h_values = dict()
            for loc, node in closed_list.items():
                # h_values[loc] = node['cost']
                self.dist_map[i,loc[0],loc[1]] = node['cost']
            # print(1)