import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from hparams import hp
import numpy as np
import time


ACTION_TO_ENCODING = [
    [],
    [],
    [2],
    [1],
    [3],
    [0],
    [1, 2],
    [2, 3],
    [0, 1],
    [0, 3],
    [2],
    [1],
    [3],
    [0],
    [1, 2],
    [2, 3],
    [0, 1],
    [0, 3]
]


def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out


def discr_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def weights_init(m):
    # TODO Use whatever weight initialization is necessary
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
        m.weight.data.normal_(0, 0.02)
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
        m.weight.data.normal_(1, 0.02)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()

def weights_init_2(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
        nn.init.xavier_normal(m.weight.data, gain=1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
        pass
        # nn.init.xavier_normal(m.weight.data, gain=1.0)
        # m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal(m.weight.data, gain=1.0)
        m.bias.data.zero_()

class EnvModel(nn.Module):
    """Network which given an input image frame consisting of last 3 frames and action
       , predicts the subsequent frame"""
    def __init__(self, num_channels=4, dilation=1):
        super(EnvModel, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, 5, stride=1, padding=2*dilation, dilation=dilation)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=dilation, dilation=dilation)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        # self.conv6 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.bn2 = nn.BatchNorm2d(64)
        self.conv_predict = nn.Conv2d(64, 1, 3, stride=1, padding=1)

        self.apply(weights_init_2)

    def forward(self, input):
        x = F.elu(self.conv1(input))
        # x = self.bn1(x)
        x = F.elu(self.conv2(x))
        # x = self.bn2(x)
        x = F.elu(self.conv3(x))
        x = self.conv_predict(x)

        return x

class AdvModel(nn.Module):
    # Follows similar architecture to DCGAN
    def __init__(self, nc=1):
        super(AdvModel, self).__init__()
        ndf = 32

        self.conv1 = nn.Conv2d(nc, ndf, 3, 2, bias=False)
        self.conv2 = nn.Conv2d(ndf, 2*ndf, 3, 2, bias=False)
        self.conv3 = nn.Conv2d(2*ndf, 4*ndf, 3, 2, bias=False)
        self.conv4 = nn.Conv2d(4*ndf, 8*ndf, 3, 2, bias=False)
        self.conv5 = nn.Conv2d(8*ndf, 1, 2, 2, bias=False)
        self.output = nn.Sigmoid()

        self.apply(discr_weights_init)

    def forward(self, input):
        x = nn.LeakyReLU(0.2, inplace=True)(self.conv1(input))
        x = nn.LeakyReLU(0.2, inplace=True)(self.conv2(x))
        x = nn.LeakyReLU(0.2, inplace=True)(self.conv3(x))
        x = nn.LeakyReLU(0.2, inplace=True)(self.conv4(x))
        x = nn.LeakyReLU(0.2, inplace=True)(self.conv5(x))

        output = self.output(x)
        return output.view(-1, 1).squeeze(1)


class ModelFree(nn.Module):
    """Model free path for predicting the action and value"""
    def __init__(self, num_channels=3, actions=4):
        super(ModelFree, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.actor_linear = nn.Linear(hp.model_conv_output_dim, actions)
        self.embed_linear = nn.Linear(hp.model_conv_output_dim, hp.model_output_dim)

        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)

    def forward(self, inputs):
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))

        x = x.view(-1, hp.model_conv_output_dim)
        return self.actor_linear(x), self.embed_linear(x)


class I3A(nn.Module):
    def __init__(self, env_model, actions=4):
        super(I3A, self).__init__()
        self.encoder_lstm = nn.LSTM(hp.conv_output_dim, hp.encoder_output_dim)
        self.model_free = ModelFree(actions=actions)
        self.env_model = env_model

        self.lstm = nn.LSTM(hp.joint_input_dim, hp.lstm_output_dim)
        # State size represents the size of LSTM cell state
        self.state_size = hp.lstm_output_dim
        self.critic_linear = nn.Linear(hp.lstm_output_dim, 1)
        self.actor_linear = nn.Linear(hp.lstm_output_dim, actions)

        # Encoder convolution network
        self.e_conv1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)
        self.e_conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.e_conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.e_conv4 = nn.Conv2d(32, 16, 3, stride=2, padding=1)

        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih_l0.data.fill_(0)
        self.lstm.bias_hh_l0.data.fill_(0)
        self.encoder_lstm.bias_ih_l0.data.fill_(0)
        self.encoder_lstm.bias_hh_l0.data.fill_(0)

        # Set the forget gate bias to be 0
        # Note gate biases are i, f, g, o
        l = hp.lstm_output_dim
        e = hp.encoder_output_dim
        self.lstm.bias_ih_l0.data[l:2*l].fill_(0.5)
        self.lstm.bias_hh_l0.data[l:2*l].fill_(0.5)
        self.encoder_lstm.bias_ih_l0.data[e:2*e].fill_(0.5)
        self.encoder_lstm.bias_hh_l0.data[e:2*e].fill_(0.5)

        self.train()

    def forward(self, inputs, training=True):
        (input_, (hx, cx), mask) = inputs
        traj_encodings = []
        
        for i in range(hp.traj_num):
            traj_encodings += [self.sample_env(input_, hp.traj_length)]

        traj_encoding = torch.cat(traj_encodings, dim=1)
        m_free_log, model_free_encoding = self.model_free(input_)

        combined_enc = torch.cat([traj_encoding, model_free_encoding], dim=1)

        if input_.size(0) == hx.size(0):
            hx = hx * mask
            cx = cx * mask
        else:
            # Compact representation to show states
            mask = mask.view(-1, hx.size(0), 1)
            combined_enc = combined_enc.view(-1, hx.size(0), combined_enc.size(1))
            hx = hx.contiguous().view(1, hx.size(0), -1).contiguous()
            cx = cx.contiguous().view(1, cx.size(0), -1).contiguous()

        if len(hx.size()) == 2:
            hx = hx.unsqueeze(0)
            cx = cx.unsqueeze(0)
            combined_enc = combined_enc.unsqueeze(0)

            x, (hx, cx) = self.lstm(combined_enc, (hx, cx))
            x = x.view(-1, hp.lstm_output_dim)
        else:
            # We have something with masks
            if len(combined_enc.size()) == 2:
                combined_enc = combined_enc.unsqueeze(0)

            outputs = []
            for i in range(combined_enc.size(0)):
                # print(combined_enc[i:i+1].shape, (hx * mask[i], cx * mask[i]))
                _, (hx, cx) = self.lstm(combined_enc[i:i+1], (hx * mask[i], cx * mask[i]))
                outputs.append(hx)
            x = torch.cat(outputs, 0)
            x = x.view(-1, hp.lstm_output_dim)    
                            
        return self.critic_linear(x), self.actor_linear(x), m_free_log, (hx, cx)


    def sample_env(self, inputs, n, initial_action=-1):
        """Input should be nx3x50x50 tensor of the past 3 states observed"""
        frames = list(torch.split(inputs, 1, 1))
        batch_size = inputs.size(0)

        # First generate our list of observations
        for i in range(n):
            inputs = torch.cat(frames[i:i+3], dim=1)
            if i==0 and initial_action!=-1:
                action_encoding = torch.Tensor([[initial_action]])
                action_encoding = action_encoding.cpu().numpy()
            else:
                actions, _ = self.model_free(inputs)
                # For stability, detach the gradient calculated for actions
                actions = F.softmax(actions, dim=1)
                # print(actions.shape)
                actions = actions.multinomial(actions.shape[0]).float()
                actions = actions.detach()
                action_encoding = actions.data.cpu().numpy()

            action_conv = torch.zeros(batch_size, 4, 50, 50).float()

            for j in range(action_encoding.shape[0]):
                fills = ACTION_TO_ENCODING[int(action_encoding[j][0])]
                for ind in fills:
                    action_conv[j,ind].fill_(2 if action_encoding[j][0] <= 9 else 1)
            action_conv = Variable(action_conv)

            conv_input = torch.cat([inputs, action_conv], 1)

            o_frame = self.env_model(conv_input)
            frames.append(o_frame)

        # Remove the first 3 input frames
        frames = frames[3:]
        frames_rev = frames[::-1]

        # Each frame should now be nx1x50x50
        frame_seq = torch.cat(frames_rev, dim=1)
        # Now each frame is size n x traj_len x 50 x 50
        frame_seq = frame_seq.view(-1, 1, 50, 50)
        # Feed through convolutional networks
        x = F.elu(self.e_conv1(frame_seq))
        x = F.elu(self.e_conv2(x))
        x = F.elu(self.e_conv3(x))
        x = F.elu(self.e_conv4(x))

        frame_seq = x.view(batch_size, hp.traj_length, -1)
        frame_seq = frame_seq.transpose(0, 1)
        _, (_, cx) = self.encoder_lstm(frame_seq)

        cx = cx.transpose(0, 1).contiguous()
        batch_size = cx.size(0)

        return cx.view(-1, hp.encoder_output_dim)
