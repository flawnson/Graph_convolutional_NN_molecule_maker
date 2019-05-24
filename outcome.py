import torch
import os

import numpy as np
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from scipy.sparse import random
from scipy import stats
from numpy.random import normal
from sklearn import model_selection

# Define variables for the document below
training_size = .70
validation_size = .15
testing_size = .15

def yielder():
        for subdir, dirs, files in os.walk(r"C:\Users\Flawnson\Documents\Project Seraph & Cherub\Project Outcome\datasets\processed"):
                for file in files:
                        filepath = subdir + os.sep + file
                        if filepath.endswith(".pth"):
                                datasets = torch.load(str(filepath))

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

def splitter(datasets):
        assert training_size >= 0 and training_size <= 1, "Invalid training set fraction"

        train, tmp = model_selection.train_test_split(datasets, train_size=training_size)
        val, test = model_selection.train_test_split(tmp, train_size=0.5) # This splits the tmp value

        return train, val, test

train, val, test = splitter(list(yielder()))