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
        seed = seed
    ) 
    return env

def load_actions(model_path, map_name, seed):
    env = make_environment(map_name, seed)
    # env = ResizeWrapper(env)
    # env = NormalizeWrapper(env)
    env = ImgWrapper(env)  # to make the images from 160x120x3 into 3x160x120
    # env = ActionWrapper(env)
    # env = DtRewardWrapper2(env)
    env = DiscreteWrapper_9(env)

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
    step_evaluation_frequency = 1
    action = 0
    action_count = 0
    actions = []
    rewards = 0
    total_speed = 0

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
            actions.append(action)
            vector = env.action(action)
            total_speed += vector[0]

            # env.render()

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

    for seed in seeds["map1"]:
        seed_results = []
        for file in os.listdir(directory):
            filename = os.fsdecode(file)

            if filename.startswith("2021-04-19_13-53-59_a3c-disc-duckie_a9-") and filename.endswith(".pth"): 
                actions, rewards, average_speed = load_actions(directory+filename, map_name, seed)
                seed_results.append(f"Model:{filename} Rewards:{rewards:.2f} Average Speed:{average_speed:.4f}")

        print(f"Seed: {seed}")
        for result in seed_results:
            print(result)
