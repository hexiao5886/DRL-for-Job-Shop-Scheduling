import torch
import torch.nn as nn

# a function that return a mlp class given sizes as a list

def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

# use case of mlp function

sizes = [2, 3, 4]

mlp_model = mlp(sizes)

net = mlp(sizes)

print(net)

print(mlp_model)


