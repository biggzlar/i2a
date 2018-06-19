import gym
import os, sys
import numpy as np
import argparse

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

from envs import make_env
from networks import EnvModel, I3A


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', default = '')
    parser.add_argument('--env-path', default = '')
    parser.add_argument('--render', default = False)
    parser.add_argument('--n_experiments', default = 8)

    args = parser.parse_args()

    # create environment
    env_ = make_env('Frostbite-v0', 0, 0, eval=True)()

    # load environment model
    env_path = args.env_path
    env_model = EnvModel(num_channels=7)
    if os.path.isfile(env_path):
        env_model_params = torch.load(env_path)
        env_model.load_state_dict(env_model_params['model'])
        print('Environment model loaded.')
    else:
        print('Running without trained environment model.')

    # load agent
    model_path = args.model_path
    model = I3A(env_model=env_model, actions=env_.action_space.n)
    if os.path.isfile(model_path):
        snapshot = torch.load(model_path)
        model.load_state_dict(snapshot['model'])
        print('Agent model loaded.')
    else:
        print('Running without trained agent model.')


    # play episodes 
    avg_score = 0
    n_episodes = 1000
    for i in range(n_episodes):
        state = env_.reset()
        model.eval()
        score = 0
        hx, cx = Variable(torch.zeros(1, 1, model.state_size)), Variable(torch.zeros(1, 1, model.state_size))
        mask = Variable(torch.zeros(1, 1, 1))
        while True:
            state = Variable(torch.from_numpy(state).unsqueeze(0).float())
            env_.render()

            # Action logits
            action_logit = model.forward((state, (hx, cx), mask))[1]
            action_probs = F.softmax(action_logit)
            actions = action_probs.multinomial()
            action = actions[0, 0]
            state, reward, done, _ = env_.step(action.data[0])
                
            score += reward
            if done:
                break

        avg_score += score
        print(' * Average Score of {}'.format((avg_score/(i+1))))