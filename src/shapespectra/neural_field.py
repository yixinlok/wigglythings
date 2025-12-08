'''
Snippet from colab
Original file is located at
    https://colab.research.google.com/drive/1bdeap8sxdQXVUGOM1YUbO0ZhRnHxMPsR

'''

import torch
import torch.nn as nn

"""Now we define our neural field, a continuous function that maps both spatial coordinates and shape-space parameters to field values.

Since the airplane geometry changes with the input code, its discretization (number and layout of sample points) also changes across shapes.
To stay discretization-agnostic, we use a neural field representation (an MLP) that learns a continuous function over coordinates instead of relying on a fixed mesh grid.

This allows us to represent and differentiate across varying shape samplings in a consistent way, enabling smooth eigenanalysis across the entire shape space.
"""

# Define the MLP model

class MLP_Code(nn.Module):
    def __init__(self, input_size = 3, geo_size = 1, hidden_size1 = 128, hidden_size2 = 128, output_size = 1):
        super(MLP_Code, self).__init__()

        self.layer1 = nn.Linear(input_size * 14 + geo_size, hidden_size1)
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.layer3 = nn.Linear(hidden_size1, hidden_size2)
        self.layer4 = nn.Linear(hidden_size1, hidden_size2)
        self.layer5 = nn.Linear(hidden_size2, output_size)
        self.dim = input_size



    def forward(self, x):

        geo = x[:,self.dim:].clone()
        x0 = x[:,:self.dim].clone()

        # positional encoding
        x1 = torch.sin(1 * x0)
        x2 = torch.cos(1 * x0)

        x3 = torch.sin(2 * x0)
        x4 = torch.cos(2 * x0)

        x5 = torch.sin(4 * x0)
        x6 = torch.cos(4 * x0)

        x7 = torch.sin(8 * x0)
        x8 = torch.cos(8 * x0)

        x9 = torch.sin(16 * x0)
        x10 = torch.cos(16 * x0)

        x11 = torch.sin(32 * x0)
        x12 = torch.cos(32 * x0)

        x13 = torch.sin(64 * x0)
        x14 = torch.cos(64 * x0)

        x = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, geo), axis = 1)

        x = torch.sin(self.layer1(x))
        x = torch.sin(self.layer2(x))
        x = torch.sin(self.layer3(x))
        x = torch.sin(self.layer4(x))
        x = self.layer5(x)


        return x
    