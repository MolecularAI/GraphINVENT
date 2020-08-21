# load general packages and functions
from torch.utils.tensorboard import SummaryWriter

# load program-specific functions
from parameters.constants import constants as C

# defines the tensorboard writer



writer = SummaryWriter(log_dir=C.tensorboard_dir, flush_secs=10)
