# Imagination-augmented agents for deep reinforcement learning
Credit: [yilundu/imagination_augmented_agents](https://github.com/yilundu/imagination_augmented_agents)

Continuation of the credited project, based on [Imagination Augmented Agents for Deep Reinforcement Learning](https://arxiv.org/abs/1707.06203). This version is implemented with A2C/PPO, as opposed to A3C.

## Setup

Tested with Python3.6 on Ubuntu 18.04 and macOS 10.13.4.

### Prerequisites
```bash
    $ sudo apt-get install python-mpi4py

    # you might want to use torch==0.4.0 on Ubuntu
    $ pip install torch==0.3.1 gym scikit-image baselines opencv-python
    
    # recent changes might require the following
    $ pip install 'gym[atari]'
    
    # if you want to train your own agent, the following is required
    $ pip install torchvision h5py
```

Should you encounter issues with baselines, [check here for additional requirements](https://github.com/openai/baselines)

## Training

Let it be known, that training this agent takes a significant amount of time and resources. You will require upwards of 60GB Ram or alternatively a large enough swapfile.

Several steps are required to train an I2A, first we need to train the environment model! 

1. Enter the `'/data'` folder and run the `gen_data.py` script (this will create a number transitions in the real environment to train the model on):

```bash
    $ cd imagination_augmented_agents/data/
    $ python gen_data.py
    
    (...)
    263   100012
    (100000, 4, 50, 50)
    (100000, 50, 50)
    Time elapsed: 993.1536860466003
```

2. Navigate to the `'/train'` folder and run the `pretrain_env.py` script to train the environment on the newly created data:

```bash
    $ cd ../train/
    $ python pretrain_env.py --exp out/path/to/env/model
```

3. Once the environment is ready we can start running the actual training:

```bash
    $ python train.py --exp out/path/to/i2a/model --env Frostbite-v0 --env-path env/model/path --snapshot 100 --eval 20
```

## Running an Agent

To run a trained agent, simply run the following script:

```bash
    # example env_model: exp/env.pth ; example i2a: exp/1140-nn.pth
    $ python demo_agent.py --model-path path/to/i2a --env-path path/to/env/model
```

Right now, the only way to render the agents activity is to edit the script itself. Navigate to line 60 and comment/uncomment the line `env_.render()`.
