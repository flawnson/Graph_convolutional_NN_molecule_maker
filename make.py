import os.path as osp
import features

import torch

from torch_geometric.data import Data, Dataset

class MyOwnDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(MyOwnDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return ["single file"]

    @property
    def processed_file_names(self):
        return ["single file"]

    def __len__(self):
        return len(self.processed_file_names)

    def download(self):
        self = self.raw_dir

    def process(self):
        i = 0
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            for x, e_index, e_attr in list(zip(features.return_node_attr(features.data_instance),
                                               features.return_edge_index(features.data_instance),
                                               features.return_edge_attr(features.data_instance))):
                data = Data(x=x, edge_index=e_index, edge_attr=e_attr)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(i)))
                i += 1

d = MyOwnDataset(root=r"C:\Users\Flawnson\Documents\Project Seraph & Cherub\Project Outcome\datasets")