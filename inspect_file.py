# import torch
# import torchvision

# import torch.optim as optim
# import torchvision.transforms as transforms
# from torch.utils.data import TensorDataset, DataLoader


# def load_and_inspect(file_path):
#     data = torch.load(file_path)
#     # Printing the type and shape of the loaded data
#     print("Type of loaded data:", type(data))
#     if isinstance(data, torch.Tensor):
#         print("Shape of the tensor:", data.shape)
#     elif isinstance(data, dict):
#         print("Keys in the dictionary:", data.keys())
#     else:
#         print("Data is neither a tensor nor a dictionary.")
#     return

# images_path = "./weights_best.pt" 
# loaded_data = load_and_inspect(images_path)


# import os
# import argparse
# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import wandb
import copy
import random


print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))
print(torch.__version__)
print(torch.cuda.device_count())
cuda_available = torch.cuda.is_available()
print(cuda_available)

# print(torch.cuda.get_arch_list())