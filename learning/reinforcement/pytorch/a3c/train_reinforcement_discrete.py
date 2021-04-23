import argparse
import logging
import datetime
import os
import sys
import numpy as np

import sys
sys.path.append(os.path.join(os.getcwd(), "gym_duckietown"))
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "learning"))
sys.path.append(os.path.join(os.getcwd(), "/learning/utils_global"))

# Duckietown Specific
from learning.reinforcement.pytorch.a3c import a3c_cnn_discrete_gru as a3c
from learning.reinforcement.pytorch.a3c import CustomOptimizer
from learning.reinforcement.pytorch.utils import seed
from learning.utils.env import launch_env
from learning.utils.wrappers import NormalizeWrapper, ImgWrapper, \
    DtRewardWrapper2, ActionWrapper, ResizeWrapper, DiscreteWrapper_9, DiscreteWrapper_9_custom, DiscreteWrapper_9_map5

# PyTorch
import torch
import torch.multiprocessing as mp

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _train(args):
    # Ensure that multiprocessing works properly without deadlock...
    if sys.version_info[0] > 2:
        mp.set_start_method('spawn')

    env = launch_env(args.map_name)
    # env = ResizeWrapper(env)
    # env = NormalizeWrapper(env)
    env = ImgWrapper(env)  # to make the images from 160x120x3 into 3x160x120
    # env = ActionWrapper(env)
    env = DtRewardWrapper2(env)
    env = DiscreteWrapper_9_map5(env)

    # Set seeds
    seed(args.seed)

    shape_obs_space = env.observation_space.shape  # (3, 120, 160)
    shape_action_space = env.action_space.n  # 3

    print("Initializing Global Network")
    global_net = a3c.Net(channels=1, num_actions=shape_action_space)
    global_net.share_memory()  # share the global parameters in multiprocessing
    optimizer = CustomOptimizer.SharedAdam(global_net.parameters(), lr=args.learning_rate)
    info = {k: torch.DoubleTensor([0]).share_memory_() for k in
            ['run_epr', 'run_loss', 'episodes', 'frames']}
    info['ep_rewards'] = []
    info['ep_losses'] = []
    info['timestamps'] = []

    if args.model_file is not None:
        cwd = os.getcwd()
        filepath = os.path.join(cwd, args.model_dir, args.model_file)
        checkpoint = torch.load(filepath)
        global_net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        info = checkpoint['info']
        print('Loaded model:', args.model_file)

    print("Instantiating %i workers" % args.num_workers)

    workers = [
        a3c.Worker(global_net, optimizer, args, info, identifier=i)
        for i in range(args.num_workers)]

    print("Start training...")

    interrupted = False

    for w in workers:
        w.daemon = True
        w.start()

    try:
        [w.join() for w in workers]
    except KeyboardInterrupt:
        [w.terminate() for w in workers]
        interrupted = True

    if not interrupted or args.save_on_interrupt:
        print("Finished training.")

        if args.save_models:
            cwd = os.getcwd()
            filedir = args.save_dir

            try:
                os.makedirs(os.path.join(cwd, filedir))
            except FileExistsError:
                # directory already exists
                pass

            filename = args.start_date + args.experiment_name + '-final.pth'
            path = os.path.join(cwd, filedir, filename)

            torch.save({
                'model_state_dict': global_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'info': info
            }, path)

            # torch.save(global_net.state_dict(), path)
            print("Saved model to:", path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=1, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--max_steps", default=10_000_000, type=int)  # Max time steps to run environment for
    parser.add_argument("--steps_until_sync", default=20, type=int)  # Steps until nets are synced
    parser.add_argument("--learning_rate", default=1e-4, type=float)  # Learning rate for the net
    parser.add_argument("--gamma", default=0.99, type=float)  # Discount factor
    parser.add_argument("--num_workers", default=8, type=int)  # Number of processes to spawn
    parser.add_argument("--save_models", default=True)  # Whether or not models are saved
    parser.add_argument("--save_frequency", default=1_000, type=int)  # Episodes to wait before saving the model
    parser.add_argument('--model_dir', type=str, default='models')  # Name of the directory where the models are saved
    parser.add_argument('--model_file', type=str, default=None)  # Name of the model to load
    parser.add_argument('--graphical_output', default=False)  # Whether to render the observation in a window
    parser.add_argument('--experiment_name', type=str, default='a3c-disc-duckie_a9')
    parser.add_argument('--start_date', type=str, default=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_"))
    parser.add_argument('--env', default=None)
    parser.add_argument('--save_on_interrupt', default=True)
    parser.add_argument('--map_name', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)

    args = parser.parse_args()

    # Pretrained models for various models
    map1_model_dir = "./models/map1/"
    map1_model = "2021-04-19_13-53-59_a3c-disc-duckie_a9-final.pth"

    map2_model_dir = "./models/map2/"
    map2_model = "2021-04-20_17-23-54_a3c-disc-duckie_a9-final.pth"

    map3_model_dir = "./models/map3/"
    map3_model = "2021-04-21_15-42-23_a3c-disc-duckie_a9-final.pth"

    map4_model_dir = "./models/map4/"
    map4_model = "2021-04-22_01-27-35_a3c-disc-duckie_a9-final.pth"

    map5_model_dir = "./models/map5/"
    map5_model = "2021-04-22_18-05-09_a3c-disc-duckie_a9-final.pth"

    # Select suitable pretrained model for transfer learning or further training
    model_dir = map5_model_dir
    model_file = map5_model
    model_steps = torch.load(model_dir + model_file)['info']['frames'][0]

    map_name = "map5"
    save_dir = "./models/map5_sharper_turn/"
    max_steps = model_steps + 20_000_000

    # Manually change args to reflect choices
    args.max_steps = max_steps
    args.model_dir = model_dir
    args.model_file = model_file
    args.map_name = map_name
    args.save_dir = save_dir

    # Reduce workers for training on laptop
    args.num_workers = 8

    _train(args)
