import torch
import os

import numpy as np
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from scipy.sparse import random
from scipy import stats
from numpy.random import normal

def yielder():
        for subdir, dirs, files in os.walk(r"C:\Users\Flawnson\Documents\Project Seraph & Cherub\Project Outcome\datasets\processed"):
                for file in files:
                        filepath = subdir + os.sep + file
                        if filepath.endswith(".pt"):
                                datasets = torch.load(str(filepath))
                                print(datasets)
                                yield datasets

def get_network_params():
        datasets = list(yielder())
        datapoint = datasets[0]
        num_attr = datapoint.num_features
        list_num_atoms = []

        for datapoint in datasets:
                list_num_atoms.append(datapoint.num_nodes)
        # Other details are defined and included here
        return num_attr, list_num_atoms

num_attr, list_num_atoms = get_network_params()

class CustomRandomState(np.random.RandomState):
        def randint(self, k):
                i = np.random.randint(k)
                return i - i % 2
np.random.seed(12345)
rs = CustomRandomState()
rvs = stats.poisson(25, loc=10).rvs

def noise_maker(number_of_attributes, list_of_number_of_atoms):

        for num_atoms in list_of_number_of_atoms:
                n = random(num_attr, num_atoms, density=1, random_state=rs, data_rvs=rvs)
        return n.A

print(noise_maker(num_attr, list_num_atoms))