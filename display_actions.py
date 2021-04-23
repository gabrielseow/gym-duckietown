import os
import numpy as np
from gym_duckietown.envs import DuckietownEnv
from generate_model_results import load_actions, make_environment

def display_actions_from_file(map_name, seed, file_path):
    env = make_environment(map_name, seed)

    obs = env.reset()
    env.render()

    total_reward = 0

    # please remove this line for your own policy
    actions = np.loadtxt(file_path, delimiter=',')

    for (speed, steering) in actions:

        obs, reward, done, info = env.step([speed, steering])
        total_reward += reward
    
        print('Steps = %s, Timestep Reward=%.3f, Total Reward=%.3f' % (env.step_count, reward, total_reward))

        env.render()

    print("Total Reward", total_reward)

def write_actions_to_file(actions, directory, file_path):
    
    action_mapping = {
    # Left
        0:[0.35, +1.0],
        1:[0.3725, +.75],
        2:[0.395, +.5],
        3:[0.4175, +.25],
    # Right
        4:[0.35, -1.0],
        5:[0.3725, -.75],
        6:[0.395, -.5],
        7:[0.4175, -.25],
    # Forward
        8:[0.44, 0.0]
    }
    
    os.makedirs(directory, exist_ok=True)
    actions = np.array([action_mapping[action[0][0]] for action in actions])

    # dump the controls using numpy
    np.savetxt(file_path, actions, delimiter=',') 

if __name__ == '__main__':
    model_dir = "models\\map5\\"
    model_file_path = model_dir + "2021-04-22_18-05-09_a3c-disc-duckie_a9-141000.0.pth"

    map_name = 'map5'
    seed = 1

    actions_dir = "results\\map5\\"
    actions_file_path = actions_dir + "map5_seed1.txt"

    ### Generate actions using model and dump actions to txt file

    #actions, rewards, average_speed, direction, turns = load_actions(model_file_path, map_name, [seed], re_orientate=True, save_actions=True)[0]
    print(load_actions(model_file_path, map_name, [seed], re_orientate=False, save_actions=True)[0])

    print(f"Rewards:{rewards} Ave Speed:{average_speed}")

    write_actions_to_file(actions, actions_dir, actions_file_path)

    ### Load actions from txt file and display using simulator
    #display_actions_from_file(map_name, seed, actions_file_path)
