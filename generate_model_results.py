import os
import torch
import torch.nn.functional as F
import argparse
import numpy as np
from gym_duckietown.envs import DuckietownEnv
from learning.reinforcement.pytorch.a3c import a3c_cnn_discrete_gru as a3c
from learning.utils.wrappers import NormalizeWrapper, ImgWrapper, \
    DtRewardWrapper2, ActionWrapper, ResizeWrapper, DiscreteWrapper_9, DiscreteWrapper_9_testing


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

def load_actions(model_path, map_name, seeds, re_orientate = True, save_actions = False):

    FULL_TURN = 135
    TURN_LEFT = 9
    TURN_RIGHT = 10

    #env = make_environment(map_name, 1)
    #env = ImgWrapper(env)  # to make the images from 160x120x3 into 3x160x120
    #env = DiscreteWrapper_9(env)

    #shape_obs_space = env.observation_space.shape  # (3, 120, 160)
    #shape_action_space = env.action_space.n  # (2,)

    shape_obs_space = (3, 480, 640)
    shape_action_space = 9

    #env.close()
    #del env

    # Load model
    cwd = os.getcwd()
    path = os.path.join(cwd, model_path)
    print('Loading model from:', path)

    checkpoint = torch.load(path)
    global_net = a3c.Net(channels=1, num_actions=shape_action_space)
    global_net.load_state_dict(checkpoint['model_state_dict'])
    global_net.eval()

    results = []

    for seed in seeds:

        if re_orientate:
            # Create separate environment to search for optimal orientation
            env_dup = make_environment(map_name, seed)
            env_dup = ImgWrapper(env_dup)  # to make the images from 160x120x3 into 3x160x120
            env_dup = DiscreteWrapper_9_testing(env_dup)

            state_dup = torch.tensor(preprocess_state(env_dup.reset()))
            done_dup = True
            
            best_val = 0
            turns = 0

            v1 = 0
            v2 = 0

            hx_dup = torch.zeros(1, 256)

            # Initialize values for adjacent views
            with torch.no_grad():
                for i in range(2):
                    hx_dup = hx_dup.detach()
                
                    # Inference
                    value_dup, _, hx_dup = global_net.forward((state_dup.view(-1, 1, 80, 80), hx_dup))
                    if i == 0:
                        v1 = value_dup
                    elif i == 1:
                        v2 = value_dup

                    # Perform action
                    state_dup, _, done_dup, _ = env_dup.step(TURN_LEFT)
                    state_dup = torch.tensor(preprocess_state(state_dup))

                    if done_dup:
                        assert False, "Error: done flag is True during initialization"

            # Search for ideal orientation
            with torch.no_grad():
                for i in range(FULL_TURN):
                    hx_dup = hx_dup.detach()

                    # Inference
                    value_dup, _, hx_dup = global_net.forward((state_dup.view(-1, 1, 80, 80), hx_dup))

                    adjusted_value = 0.5*(value_dup+v1) + v2
                    if adjusted_value > best_val:
                        best_val = adjusted_value
                        turns = i+1
                    v1 = v2
                    v2 = value_dup

                    # Perform action
                    state_dup, _, done_dup, _ = env_dup.step(TURN_LEFT)
                    state_dup = torch.tensor(preprocess_state(state_dup))

                    if done_dup:
                        assert False, "Error: done flag is True during searching"

            if turns > FULL_TURN // 2:
                direction = TURN_RIGHT
                turns = FULL_TURN - turns
            else:
                direction = TURN_LEFT

            env_dup.close()

            del env_dup
            del state_dup
            del hx_dup

        env = make_environment(map_name, seed)
        env = ImgWrapper(env)  # to make the images from 160x120x3 into 3x160x120

        if re_orientate:
            env = DiscreteWrapper_9_testing(env)
        else:
            env = DiscreteWrapper_9(env)

        steps_until_new_action = 0
        step_evaluation_frequency = 1

        state = torch.tensor(preprocess_state(env.reset()))
        done = True

        action = 0
        action_count = 0
        actions = []
        rewards = 0
        total_speed = 0
        crashed = False

        if re_orientate:
            # Execute turns on real environment
            with torch.no_grad():
                for i in range(turns):
                    # Perform action
                    state, reward, done, _ = env.step(direction)
                    state = torch.tensor(preprocess_state(state))

                    # Track all actions taken
                    rewards += reward
                    action_count += 1
                    if save_actions:
                        actions.append(direction)
                
                    #env.render()

                    if reward == -1000:
                        assert False, "Error: crashed during re_orientate"
                    if done:
                        assert False, "Error: done flag is True during re_orientate"
        
        hx = torch.zeros(1, 256)

        while True:
            with torch.no_grad():
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

                # Abort evaluation early if robot crashes into wall
                if reward == -1000:
                    crashed = True
                    break

                # Track all actions taken
                rewards += reward
                action_count += 1
                total_speed += env.action(action)[0]
                if save_actions:
                    actions.append(action[0][0])

                #env.render()

                if done:
                    #state = torch.tensor(preprocess_state(env.reset()))
                    assert action_count == MAX_STEPS, "Error: done flag is True before MAX_STEPS reached"
                    break
        if crashed: 
            results.append([None, rewards])
        elif re_orientate:
            results.append([actions, rewards, (total_speed/MAX_STEPS), direction, turns])
        else:
            results.append([actions, rewards, (total_speed/MAX_STEPS)])

        env.close()

        del env
        del state
        del hx

    return results

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

    map_seeds = {
        "map1": [2, 3, 5, 9, 12],
        "map2": [1, 2, 3, 5, 7, 8, 13, 16],
        "map3": [1, 2, 4, 8, 9, 10, 15, 21],
        "map4": [1, 2, 3, 4, 5, 7, 9, 10, 16, 18],
        "map5": [1, 2, 4, 5, 7, 8, 9, 10, 16, 23]
    }

    directory = "models\\map3\\"
    map_name = "map3"
    re_orientate = True
        
    # Prefix for individual models to evaluate
    selected_models = [
        #"2021-04-24_01-41-59_a3c-disc-duckie_a9-15",
        #"2021-04-24_01-41-59_a3c-disc-duckie_a9-16",
        "2021-04-24_01-41-59_a3c-disc-duckie_a9-175",
        "2021-04-24_01-41-59_a3c-disc-duckie_a9-176",
        #"2021-04-24_01-41-59_a3c-disc-duckie_a9-18",
        "2021-04-24_01-41-59_a3c-disc-duckie_a9-19",
        "2021-04-24_01-41-59_a3c-disc-duckie_a9-final",
    ]
    only_print_selected = True

    # Prefix for general models to evaluate
    model_prefix = [
        "2021-04-22_18-08-07_a3c-disc-duckie_a9"
    ]

    model_prefix = model_prefix[0]

    seeds = map_seeds[map_name]
    seeds = [21]

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        seed_results_string = []

        # If evaluating only selected models
        if only_print_selected and any(filename.startswith(model) and filename.endswith(".pth") for model in selected_models):
            seed_results = load_actions(directory+filename, map_name, seeds, re_orientate = re_orientate)
        # If evaluating all models
        elif not only_print_selected and filename.startswith(model_prefix) and filename.endswith(".pth"): 
            seed_results = load_actions(directory+filename, map_name, seeds, re_orientate = re_orientate)
        else:
            continue
        
        header = f"Model: {filename} Re-orientate: {re_orientate}"
        seed_results_string.append(header)
        for i in range(len(seed_results)):
            seed = seeds[i]
            result = seed_results[i]
            if result[0] is None:
                rewards = result[1]
                seed_results_string.append(f"Seed:{seed} Crashed Rewards:{rewards:.2f}")
            elif re_orientate:
                actions, rewards, average_speed, direction, turns = result
                direction_string = "LEFT" if direction == 9 else "RIGHT"
                seed_results_string.append(f"Seed:{seed} Rewards:{rewards:.2f} Average Speed:{average_speed:.4f} Direction:{direction_string} Turns:{turns}")
            else:
                actions, rewards, average_speed = result
                seed_results_string.append(f"Seed:{seed} Rewards:{rewards:.2f} Average Speed:{average_speed:.4f}")

        # Open results file
        results_file = open(directory + model_prefix + ".txt", mode="a")

        # Print evaluation results and store in results file
        for result in seed_results_string:
            print(result)
            results_file.write(result+'\n')
        results_file.close()
