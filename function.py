import torch


def weight_scaling_init(layer):
    w = layer.weight.detach()
    alpha = 10.0 * w.std()
    layer.weight.data /= torch.sqrt(alpha)
    layer.bias.data /= torch.sqrt(alpha)
