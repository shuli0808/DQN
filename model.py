import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, n_in=[3,210,160], conv_channels=[32, 64, 64],
                 conv_kernels=[8, 4, 3], conv_strides=[4, 2, 1],  n_atoms=51, 
                 Vmin=-10., Vmax= 30, n_fc = [128, 256],n_out = 6):
        super(Model, self).__init__()
        #paramters from paper
        self.n_in = n_in
        # Number of actions depending on the game
        self.n_out = n_out
        self.n_atoms = n_atoms
        self.Vmin = Vmin
        self.Vmax = Vmax

        c0 = n_in[0]
        h0 = n_in[1]

        self.conv_layers = []
        for c, k, s in zip(conv_channels, conv_kernels, conv_strides):
            # append nn.Conv2d with kernel size k, stride s
            self.conv_layers.append(nn.Conv2d(c0, c, kernel_size = k, stride = s))
            # append nn.ReLU layer
            self.conv_layers.append(nn.ReLU())

            h0 = int(float(h0-k) / s + 1)
            c0 = c
        self.conv_layers = nn.Sequential(*self.conv_layers)

        # self.fc = nn.Sequential(
        #     nn.Linear(self.input_dim[0], 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, self.output_dim * self.n_atoms)
        # )

        self.fc_layers = []
        h0 = h0 * h0 * conv_channels[-1]
        for i, h in enumerate(n_fc):
            # append Linear and ReLU layers
            self.fc_layers.append(nn.Linear(h0, h))            
            self.fc_layers.append(nn.ReLU())
            h0 = h
        
        self.fc_layers.append(nn.Linear(h, self.n_out*self.n_atoms))
        self.fc_layers.append(nn.Softmax(dim=1))
        self.fc_layers = nn.Sequential(*self.fc_layers)


    def forward(self, x, head=None):
        x = x.float() / 256

        # feed x into the self.conv_layers
        x = self.conv_layers(x)
        # (flatten) reshape x into a batch of vectors
        x = x.view(x.size(0), -1)
        # feed x into the self.fc_layers
        x = self.fc_layers(x)

        return x