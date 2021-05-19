import torch
import os

seed = 2021

# 应对DataLoader切分文件过多出现错误
torch.multiprocessing.set_sharing_strategy('file_system')
