import os
import numpy as np
from gym_duckietown.envs import DuckietownEnv
from generate_model_results import load_actions, make_environment

def display_actions_from_file(map_name, seed, file_path):
    env = DuckietownEnv(
        map_name = map_name,
        domain_rand = False,
        draw_bbox = False,
        max_steps = MAX_STEPS,
        seed = seed
    )

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
        8:[0.44, 0.0],
    # Rotate on spot
        9:[0.0, +1.0],
        10:[0.0, -1.0]
    }
    
    os.makedirs(directory, exist_ok=True)

    actions = np.array([action_mapping[action] for action in actions])

    # dump the controls using numpy
    np.savetxt(file_path, actions, delimiter=',') 

MAX_STEPS = 1500

if __name__ == '__main__':
    map_name = 'map5'
    seed = 2

    model_dir = f"models\\{map_name}\\"
    file_name = [
    "2021-04-22_18-05-09_a3c-disc-duckie_a9-147000.0.pth"
    ]
    model_file_path = model_dir + file_name[0]

    actions_dir = f"results\\{map_name}\\"
    actions_file_path = actions_dir + f"{map_name}_seed{seed}.txt"

    ### Generate actions using model and dump actions to txt file

    actions, rewards, average_speed, direction, turns = load_actions(model_file_path, map_name, [seed], re_orientate=True, save_actions=True)[0]
    #actions, rewards, average_speed = load_actions(model_file_path, map_name, [seed], re_orientate=False, save_actions=True)[0]

    print(f"Rewards:{rewards} Ave Speed:{average_speed}")

    write_actions_to_file(actions, actions_dir, actions_file_path)

    ### Load actions from txt file and display using simulator
    #display_actions_from_file(map_name, seed, actions_file_path)
