# load general packages and functions
import torch

# load program-specific functions
from parameters.constants import constants as C
import util

# defines various possible loss functions


def graph_generation_loss(output, target_output):
    """ Calculated the loss using the KL divergence.

    Args:
      output (torch.Tensor) : Predicted APD tensor.
      target_output (torch.Tensor) : Target APD tensor.

    Returns:
      loss (float) : Average loss for this output.
    """
    # define activation function; note that one must use the softmax in the
    # KLDiv, never the sigmoid, as the distribution must sum to 1
    LogSoftmax = torch.nn.LogSoftmax(dim=1)

    output = LogSoftmax(output)

    # normalize the target output (as can contain information on > 1 graph)
    target_output = target_output/torch.sum(target_output, dim=1, keepdim=True)

    # define loss function and calculate the los
    criterion = torch.nn.KLDivLoss(reduction="batchmean")
    loss = criterion(target=target_output, input=output)

    return loss
