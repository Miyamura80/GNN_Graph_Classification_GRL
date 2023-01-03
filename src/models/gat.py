import torch
import torch.nn.functional as F
from torch.nn import ModuleList, BatchNorm1d
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GATConv
from torch_scatter import scatter_max, scatter_mean


class NetGAT(torch.nn.Module):
    def __init__(
        self,
        num_features,
        num_classes,
        emb_sizes=None,
        drpt_prob=0.5,
        scatter="max",
        device="cpu",
    ):
        super(NetGAT, self).__init__()
        if emb_sizes is None:  # Python default handling for mutable input
            emb_sizes = [32, 64, 64]  # The 0th entry is the input feature size.
        self.num_features = num_features
        self.emb_sizes = emb_sizes
        self.num_layers = len(self.emb_sizes) - 1
        self.drpt_prob = drpt_prob
        self.scatter = scatter
        self.device = device

        self.initial_mlp_modules = ModuleList(
            [
                Linear(num_features, emb_sizes[0]).to(device),
                BatchNorm1d(emb_sizes[0]).to(device),
                ReLU().to(device),
                Linear(emb_sizes[0], emb_sizes[0]).to(device),
                BatchNorm1d(emb_sizes[0]).to(device),
                ReLU().to(device),
            ]
        )
        self.initial_mlp = Sequential(*self.initial_mlp_modules).to(device)
        self.initial_linear = Linear(emb_sizes[0], num_classes).to(device)

        gat_layers = []
        linears = []
        for i in range(self.num_layers):
            in_channel = emb_sizes[i]
            out_channel = emb_sizes[i + 1]
            gat_layer = GATConv(in_channels=in_channel, out_channels=out_channel).to(
                device
            )
            gat_layers.append(gat_layer)
            linears.append(Linear(emb_sizes[i + 1], num_classes).to(device))

        self.gat_modules = ModuleList(gat_layers)
        self.linear_modules = ModuleList(linears)

    def reset_parameters(self):
        for (name, module) in self._modules.items():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        for module in self.gat_modules:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        for module in self.linear_modules:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        for module in self.initial_mlp_modules:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def pooling(self, x_feat, batch):
        if self.scatter == "max":
            return scatter_max(x_feat, batch, dim=0)[0].to(self.device)
        elif self.scatter == "mean":
            return scatter_mean(x_feat, batch, dim=0).to(self.device)
        else:
            pass

    def forward(self, data):
        x_feat = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        edge_attributes = data.edge_attributes.to(self.device)

        x_feat = self.initial_mlp(x_feat)

        out = F.dropout(
            self.pooling(self.initial_linear(x_feat), data.batch), 
            p=self.drpt_prob
        )

        for gat_layer, linear_layer in zip(self.gat_modules, self.linear_modules):
            edges = edge_index.T[edge_attributes == 1].T
            x_feat = gat_layer(x_feat, edges).to(self.device)

            out += F.dropout(
                linear_layer(self.pooling(x_feat, data.batch)),
                p=self.drpt_prob,
                training=self.training,
            )

        return out

