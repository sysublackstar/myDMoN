from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from networkx.readwrite.gml import read_gml
from torch.optim import Adam
from torch_geometric.data import Data
from tqdm import tqdm
import seaborn as sns
import metrics
from model import MyModel
from utils import convert_scipy_sparse_to_sparse_tensor
from torchviz import make_dot
plt.style.use(['science', 'ieee'])
SEED = 0
torch.manual_seed(SEED)

