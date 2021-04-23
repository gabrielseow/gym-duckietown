import os
import torch
import torch.nn.functional as F
import argparse
import numpy as np
from gym_duckietown.envs import DuckietownEnv
from learning.reinforcement.pytorch.a3c import a3c_cnn_discrete_gru as a3c
from learning.utils.wrappers import NormalizeWrapper, ImgWrapper, \
    DtRewardWrapper2, ActionWrapper, ResizeWrapper, DiscreteWrapper_9, DiscreteWrapper_9_custom


def preprocess_state(obs):
    from scipy.misc import imresize
    return imresize(obs.mean(2), (80, 80)).astype(np.float32).reshape(1, 80, 80) / 255.

def make_environment(map_name, seed):
    env = DuckietownEnv(
        map_name = map_name,
        domain_rand = False,
        draw_bbox = False,
        max_steps = MAX_STEPS,
        # Experiment
        #accept_start_angle_deg=4,  # start close to straight
        seed = seed
    ) 
    return env

def load_actions(model_path, map_name, seed, save_actions):
    env = make_environment(map_name, seed)
    # env = ResizeWrapper(env)
    # env = NormalizeWrapper(env)
    env = ImgWrapper(env)  # to make the images from 160x120x3 into 3x160x120
    # env = ActionWrapper(env)
    # env = DtRewardWrapper2(env)
    env = DiscreteWrapper_9_testing(env)

    shape_obs_space = env.observation_space.shape  # (3, 120, 160)
    shape_action_space = env.action_space.n  # (2,)

    # Initialize policy

    # Load model
    cwd = os.getcwd()
    path = os.path.join(cwd, model_path)
    print('Loading model from:', path)

    checkpoint = torch.load(path)
    global_net = a3c.Net(channels=1, num_actions=shape_action_space)
    global_net.load_state_dict(checkpoint['model_state_dict'])
    #global_net.load_state_dict(checkpoint)
    global_net.eval()

    state = torch.tensor(preprocess_state(env.reset()))
    done = True

    steps_until_new_action = 0
    step_evaluation_frequency = 3
    action = 0
    action_count = 0
    actions = []
    rewards = 0
    total_speed = 0

    FULL_TURN = 135
    TURN_LEFT = 9
    TURN_RIGHT = 10

    best_val = 0
    turns = 0

    # Search for ideal orientation
    with torch.no_grad():
        for i in range(FULL_TURN):
            if done:
                hx = torch.zeros(1, 256)
            else:
                hx = hx.detach()

            # Inference
            value, _, _ = global_net.forward((state.view(-1, 1, 80, 80), hx))
            if value > best_val:
                best_val = value
                turns = i

            #action_log_probs = F.log_softmax(logit, dim=-1)
            #print(f"Val: {value} Logit: {logit} HX: {hx}")
            #print(f"Log_prob: {action_log_probs}")

            # Perform action
            state, _, done, _ = env.step(TURN_LEFT)
            state = torch.tensor(preprocess_state(state))

            # env.render()

            if done:
                state = torch.tensor(preprocess_state(env.reset()))

    if turns > FULL_TURN // 2:
        direction = TURN_RIGHT
        turns = FULL_TURN - turns
    else:
        direction = TURN_LEFT

    with torch.no_grad():
        for i in range(turns):
            if done:
                hx = torch.zeros(1, 256)
            else:
                hx = hx.detach()

            # Perform action
            state, _, done, _ = env.step(direction)
            state = torch.tensor(preprocess_state(state))

            #env.render()

            if done:
                state = torch.tensor(preprocess_state(env.reset()))

    while action_count < MAX_STEPS:
        with torch.no_grad():
            if done:
                hx = torch.zeros(1, 256)
            else:
                hx = hx.detach()

            steps_until_new_action -= 1

            # Inference
            value, logit, hx = global_net.forward((state.view(-1, 1, 80, 80), hx))
            action_log_probs = F.log_softmax(logit, dim=-1)

            if steps_until_new_action <= 0:
                # Take action with highest probability
                action = action_log_probs.max(1, keepdim=True)[1].numpy()
                steps_until_new_action = step_evaluation_frequency

            # Perform action
            state, reward, done, _ = env.step(action)
            state = torch.tensor(preprocess_state(state))

            # Track all actions taken
            rewards += reward
            action_count += 1
            if save_actions:
                actions.append(action)
            vector = env.action(action)
            total_speed += vector[0]

            #print(f"Reward: {reward:.2f}")
            #env.render()

            if done:
                state = torch.tensor(preprocess_state(env.reset()))

    return actions, rewards, (total_speed/MAX_STEPS)

MAX_STEPS = 1500

if __name__ == '__main__':

    '''
    # declare the arguments
    parser = argparse.ArgumentParser()

    # Do not change this
    parser.add_argument('--max_steps', type=int, default=1500, help='max_steps')

    # You should set them to different map name and seed accordingly
    parser.add_argument('--map-name', default='map1')
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    args = parser.parse_args()
    '''

    seeds = {
        "map1": [2, 3, 5, 9, 12],
        "map2": [1, 2, 3, 5, 7, 8, 13, 16],
        "map3": [1, 2, 4, 8, 9, 10, 15, 21],
        "map4": [1, 2, 3, 4, 5, 7, 9, 10, 16, 18],
        "map5": [1, 2, 4, 5, 7, 8, 9, 10, 16, 23]
    }

    directory = "models\\map1\\"
    map_name = "map1"
        
    # Prefix for individual models to evaluate
    model_1 = "2021-04-22_18-05-09_a3c-disc-duckie_a9-145"
    model_2 = "2021-04-22_18-05-09_a3c-disc-duckie_a9-146"
    selected_models = [
        model_1,
        model_2,
    ]
    only_print_selected = False

    # Prefix for general models to evaluate
    model_prefix = "2021-04-22_18-08-07_a3c-disc-duckie_a9"

    # Create results file
    results_file = open(directory + model_prefix + ".txt", mode="a")

    # Tested
    tested = []

    # for seed in seeds[map_name]:
    for seed in [3]:
        if seed in tested:
            continue

        seed_results = []

        for file in os.listdir(directory):
            filename = os.fsdecode(file)

            # If evaluating only selected models
            if only_print_selected and any(filename.startswith(model) for model in selected_models):
                actions, rewards, average_speed = load_actions(directory+filename, map_name, seed, save_actions=False)
                seed_results.append(f"Model:{filename} Rewards:{rewards:.2f} Average Speed:{average_speed:.4f}")
            # If evaluating all models
            elif filename.startswith(model_prefix) and filename.endswith(".pth"): 
                actions, rewards, average_speed = load_actions(directory+filename, map_name, seed, save_actions=False)
                seed_results.append(f"Model:{filename} Rewards:{rewards:.2f} Average Speed:{average_speed:.4f}")


        # Print evaluation results and store in results file
        header = f"Seed: {seed}"
        results_file.write(header+'\n')
        print(header)
        for result in seed_results:
            results_file.write(result+'\n')
            print(result)
    results_file.close()
