# environment setting
# map_size = (9, 9)
# change the map_size and tune the reward value of stay

# imitation = True
# curriculum = True
# subgoal = False
# punish = False
# oscillation = False

grad_norm = None
# closer_further = True

# post_process = False
# a_star = False


map_size = (14, 14)
num_agents = 8
action_space = 5
obs_dimension = (4,4)
switch = 5


# reward setting
move_reward = -0.075
stay_on_goal_reward = 0
stay_off_goal_reward = -0.175
collision_reward = -1
finish_reward = 5   # CAN BE LARGER
oscillation_reward = -0.3


closer_reward = -0.05
further_reward = -0.1



punish_factor = 1.2


# model setting
num_kernels = 128
# latent1_dim = 488
# if punish:
#     latent2_dim = 12
#     latent3_dim = 12
# else:
#     latent2_dim = 24
all_latent_dim = 512

# training setting

n_steps = 5

batch_size = 256
double_q = True
buffer_size = 10000
exploration_start_eps = 1.0
exploration_final_eps = 0.001
train_freq = 8

learning_starts = 3000
save_interval = 5000
target_network_update_freq = 1000*train_freq
gamma = 0.99
prioritized_replay = True
prioritized_replay_alpha = 0.6
prioritized_replay_beta0 = 0.4
dueling = True

imitation_ratio = 0.3

history_steps = 4


background = 'wide_aisle_warehouse'   # 'wide_aisle_warehouse', 'narrow_aisle_warehouse', 'free_map', 'random_map'
# parameteres to build map
if background == 'wide_aisle_warehouse':
    start_place = 2
    step_stride = 4
if background =='narrow_aisle_warehouse':
    start_place = 1
    step_stride_row = 3
    step_stride_column = 4
if background =='narrow_aisle_warehouse_6*2':
    start_place = 1
    step_stride_row = 3
    step_stride_column = 7
if background =='narrow_aisle_warehouse_9*2':
    start_place = 1
    step_stride_row = 3
    step_stride_column = 10

obstacle_density = 0.3


# curriculum learning

# if curriculum:
#     initial_dis = 5
# else:
#     initial_dis = map_size[0] + map_size[1]

final_dis = map_size[0] + map_size[1]

# about test
debug = False
episode_num = 80
show_step = 200

plot_and_show = False

# select data
file_loss = "Loss"
test_map_size = (14,14)
test_agent_num = 8

CBS_MAX_TIME = 5
# test_seed_model_3 = True

device = "cpu"