import json
from torch_geometric.data import Data
import torch
import pickle
import os
import os.path as osp
import numpy as np


def read_sc_graph(direc, file, processed_root):
    path = osp.join(direc, file + ".json")
    presaved_path = osp.join(processed_root, file + ".pre")
    if not osp.exists(presaved_path):  # The file doesn't exist
        print("making directory")
        with open(path, "r") as f:
            data = f.readlines()

            data[0] = data[0][1:] + ""
            data = [jline.strip()[:len(jline.strip())-1] for jline in data]

            graphs = [json.loads(jline) for jline in data]
            # Load Json into PyG `Data` type
            pyg_graphs = [
                map_sc_graph_to_pyg(graph, make_undirected=True, remove_dup=False)
                for graph in graphs
            ]

            if not osp.exists(processed_root):
                os.mkdir(processed_root)
            with open(presaved_path, "wb") as g:  # Save for future reference
                pickle.dump(pyg_graphs, g)
                g.close()
            f.close()
            return pyg_graphs
    else:  # Load the pre-existing pickle
        print("load preexisting")
        with open(presaved_path, "rb") as g:
            pyg_graphs = pickle.load(g)
            g.close()
        return pyg_graphs


def map_sc_graph_to_pyg(json_file, make_undirected=True, remove_dup=False):
    # Note: make_undirected makes duplicate edges, so we need to preserve edge types.
    edge_index = np.array([[g[0], g[2]] for g in json_file["graph"]]).T  # Edge Index
    edge_attributes = np.array(
        [g[1] - 1 for g in json_file["graph"]]
    )  # Edge type (-1 to put in [0, 3] range)

    if (
        make_undirected
    ):  # This will invariably cost us edge types because we reduce duplicates
        edge_index_reverse = edge_index[[1, 0], :]
        # Concat and remove duplicates
        if remove_dup:
            edge_index = torch.LongTensor(
                np.unique(
                    np.concatenate([edge_index, edge_index_reverse], axis=1), axis=1
                )
            )
        else:
            edge_index = torch.LongTensor(
                np.concatenate([edge_index, edge_index_reverse], axis=1)
            )
            edge_attributes = torch.LongTensor(
                np.concatenate([edge_attributes, np.copy(edge_attributes)], axis=0)
            )
    features = np.array(json_file["node_features"])
    features = np.concatenate((features[:, 0:11], features[:,22:36]), axis=1) 
    x = torch.FloatTensor(features)
    y = torch.FloatTensor(np.array([[int(json_file["targets"])]]).T)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attributes, y=y)


def get_dataset(args, root_dir):
    dataset_path = osp.join(root_dir, "data", args.dataset)
    sc_proc_root = osp.join(dataset_path, f"{args.dataset}_proc")

    train_graphs = read_sc_graph(
        dataset_path, "train", sc_proc_root
    )
    valid_graphs = read_sc_graph(
        dataset_path, "valid", sc_proc_root
    )
    num_feat = 25
    num_pred = 1
    return train_graphs, valid_graphs, num_feat, num_pred
