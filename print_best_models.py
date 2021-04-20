import os
import torch
import torch.nn.functional as F
import argparse
import numpy as np
from gym_duckietown.envs import DuckietownEnv
from learning.reinforcement.pytorch.a3c import a3c_cnn_discrete_gru as a3c
from learning.utils.wrappers import NormalizeWrapper, ImgWrapper, \
    DtRewardWrapper2, ActionWrapper, ResizeWrapper, DiscreteWrapper_9

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

def preprocess_state(obs):
    from scipy.misc import imresize
    return imresize(obs.mean(2), (80, 80)).astype(np.float32).reshape(1, 80, 80) / 255.

def load_actions(model_path):
    env = DuckietownEnv(
        map_name = 'map1',
        # map_name = args.map_name,
        domain_rand = False,
        draw_bbox = False,
        max_steps = args.max_steps,
        seed = 2
        # seed = args.seed
    )
    # env = ResizeWrapper(env)
    # env = NormalizeWrapper(env)
    env = ImgWrapper(env)  # to make the images from 160x120x3 into 3x160x120
    # env = ActionWrapper(env)
    env = DtRewardWrapper2(env)
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
    step_evaluation_frequency = 5
    action = 0
    action_count = 0
    actions = []
    rewards = 0

    while action_count < args.max_steps:
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

            env.render()

            if done:
                state = torch.tensor(preprocess_state(env.reset()))

    return actions, rewards

# declare the arguments
parser = argparse.ArgumentParser()

# Do not change this
parser.add_argument('--max_steps', type=int, default=1500, help='max_steps')

# You should set them to different map name and seed accordingly
parser.add_argument('--map-name', default='map5')
parser.add_argument('--seed', type=int, default=11, help='random seed')
args = parser.parse_args()

directory = "models/map1/"

for file in os.listdir(directory):
     filename = os.fsdecode(file)

     if filename.endswith(".pth"): 
        actions, rewards = load_actions(directory+filename)
        print(rewards)
        break

        


'''
obs = env.reset()
env.render()

total_reward = 0

# please remove this line for your own policy
actions = np.loadtxt('./map5_seed11.txt', delimiter=',')

for (speed, steering) in actions:

    obs, reward, done, info = env.step([speed, steering])
    total_reward += reward
    
    print('Steps = %s, Timestep Reward=%.3f, Total Reward=%.3f' % (env.step_count, reward, total_reward))

    env.render()

print("Total Reward", total_reward)

# dump the controls using numpy
np.savetxt('./map5_seed11.txt', actions, delimiter=',')
'''
