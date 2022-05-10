import argparse
import operator
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import math
from functools import reduce
from operator import mul

batch_size
x = torch.randn(batch_size, 10)
x = x - 2 * x.pow(2)
y = x.sum(1)
return x, y