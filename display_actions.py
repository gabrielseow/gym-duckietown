import numpy as np
from gym_duckietown.envs import DuckietownEnv
from print_best_models import load_actions

def display_actions(map_name, seed, file_path):
    env = DuckietownEnv(
        map_name = map_name,
        domain_rand = False,
        draw_bbox = False,
        max_steps = 1500,
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

def write_actions_to_file(actions, file_path):
    
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
    
    actions = np.array([action_mapping[action[0][0]] for action in actions])
    print(actions)

    # dump the controls using numpy
    np.savetxt(file_path, actions, delimiter=',') 

if __name__ == '__main__':
    directory = "models\\map1\\"
    filename = "2021-04-19_13-53-59_a3c-disc-duckie_a9-47000.0.pth"
    map_name = 'map1'
    seed = 12

    # Generate actions using selected model, map name and seed
    # actions, rewards = load_actions(directory+filename, map_name, seed)
    # write_actions_to_file(actions, './map1_seed12.txt')

    display_actions("map1", seed, './map1_seed12.txt')
