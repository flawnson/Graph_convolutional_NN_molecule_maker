"""This file is comprised of basic UDFs that may be useful in Project Outcome.
There are a couple possible input types for each function:
1. molecules (the raw sdf file, opened to be iterated upon)
2. atom list (the atoms that each molecule of the sdf is comprised of, which can be represented by any unique ID)
"""

import scipy
import numpy as np
import pandas as pd
import itertools

import tqdm
import torch

import os.path as osp
import rdkit.Chem as Chem
import networkx as nx

from torch_geometric.data import Data, Dataset
from mendeleev import element

suppl = Chem.SDMolSupplier('10.sdf')

smiles = open("10_rndm_zinc_drugs_clean.smi").read().splitlines()

# suppl = Chem.SDMolSupplier("qm9-100-sdf.sdf")

# targets = list(pd.read_csv("qm9-100.csv").mu)

# smiles = open("qm9-100-smiles.smi").read().splitlines()

# assert len(suppl) == len(targets) == len(smiles), "the datasets must be of the same length!"

# =================================================================================== #
"""                              GRAPH REPRESENTATION                               """
# =================================================================================== #

def get_adj_matrix_coo(molecules):
    for mol in molecules:
        adj_mat = scipy.sparse.csr_matrix(Chem.rdmolops.GetAdjacencyMatrix(mol))
        nx_graph = nx.from_scipy_sparse_matrix(adj_mat)
        coo_matrix = nx.to_scipy_sparse_matrix(nx_graph, format="coo")

        yield coo_matrix.row, coo_matrix.col

coo_adj_matrix = list(get_adj_matrix_coo(suppl))

# =================================================================================== #
"""                                GRAPH ATTRIBUTES                                 """
# =================================================================================== #

def get_num_bonds(molecules):
    for mol in suppl:
        number_of_bonds = mol.GetNumBonds()

        yield number_of_bonds

number_of_bonds = list(get_num_bonds(suppl))

# =================================================================================== #
"""                                NODE ATTRIBUTES                                  """
# =================================================================================== #

def get_atom_symbols(molecules):
    for mol in molecules:
        atoms = mol.GetAtoms()
        atom_symbols = [atom.GetSymbol() for atom in atoms]
        yield atom_symbols

def get_atom_properties(atom_list):
    for atoms in atom_list:
        atomic_number = [element(atom).atomic_number for atom in atoms]
        atomic_volume = [element(atom).atomic_volume for atom in atoms]
        atomic_weight = [element(atom).atomic_weight for atom in atoms]
        all_atom_properties = list(zip(atomic_number, atomic_volume, atomic_weight))
        yield all_atom_properties

all_atom_properties = list(get_atom_properties(get_atom_symbols(suppl)))

# TODO: get_atom_properties is generalized; adding a .specific_property to element(atom) would allow
#       access to different properties from the mendeleev package. Also note that this funciton is
#       dependant on the get_atom_symbols generator directly above it.

# =================================================================================== #
"""                                EDGE ATTRIBUTES                                  """
# =================================================================================== #

def get_num_bonds(molecules):
    for mol in suppl:
        number_of_bonds = mol.GetNumBonds()

        yield number_of_bonds

def get_bonds_info(molecules):
    for mol in suppl:
        number_of_bonds = mol.GetNumBonds()
        bond_types = [bond.GetBondTypeAsDouble() for bond in mol.GetBonds()]

        yield bond_types

bond_types = list(get_bonds_info(suppl))

# =================================================================================== #
"""                                    TARGETS                                      """
# =================================================================================== #
def get_targets(atom_list): # Same as the get_atom_properties function from node attributes section
    for atoms in atom_list:
        boiling_points = list([element(atom).boiling_point for atom in atoms])
        yield boiling_points

targets = list(get_targets(get_atom_symbols(suppl)))

def normalize(numerical_dataset):
        raw = list(itertools.chain.from_iterable(numerical_dataset))

        maximum = max(raw)
        minimum = min(raw)

        for targets in numerical_dataset:
                norm_dataset = [(target - minimum) / (maximum - minimum) for target in targets]

                return norm_dataset

norm_targets = normalize(targets)

# =================================================================================== #
"""                                 BUILD DATASETS                                  """
# =================================================================================== #

def seperator(datasets):
    for example in datasets:
        tensorized_example = torch.tensor(example, dtype=torch.float)

        yield tensorized_example

edge_index = list(seperator(coo_adj_matrix)) 
node_attr = list(seperator(all_atom_properties))
edge_attr = list(seperator(bond_types))
Y_data = list(seperator(targets))

data_instance = list(map(list, zip(node_attr, edge_index, edge_attr, Y_data)))

def return_data(zipped_data):
    for instance in data_instance:
        node_attr = instance[0]
        edge_index = instance[1]
        edge_attr = instance[2]
        target = instance[3]
        print(type(instance))
        yield node_attr, edge_index, edge_attr, target

# def data_maker(datapoints):
#         for node_attr, edge_index, edge_attr, target in datapoints:
#                 data = Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr, y=target)
#                 print(type(data))
#                 yield data
# print(list(data_maker(return_data(data_instance))))