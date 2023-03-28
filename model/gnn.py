import torch
import torch.nn as nn
import torch_geometric.nn as nng
import torch.nn.functional as F


class GraphSAGE(nn.Module):
    def __init__(self, hparams, encoder, decoder):
        super(GraphSAGE, self).__init__()

        self.nb_hidden_layers = hparams['nb_hidden_layers']
        self.size_hidden_layers = hparams['size_hidden_layers']
        self.bn_bool = hparams['bn_bool']
        self.activation = nn.ReLU()

        self.encoder = encoder
        self.decoder = decoder

        self.in_layer = nng.SAGEConv(
            in_channels=hparams['encoder'][-1],
            out_channels=self.size_hidden_layers
        )

        self.hidden_layers = nn.ModuleList()
        for n in range(self.nb_hidden_layers - 1):
            self.hidden_layers.append(nng.SAGEConv(
                in_channels=self.size_hidden_layers,
                out_channels=self.size_hidden_layers
            ))

        self.out_layer = nng.SAGEConv(
            in_channels=self.size_hidden_layers,
            out_channels=hparams['decoder'][0]
        )

        if self.bn_bool:
            self.bn = nn.ModuleList()
            for n in range(self.nb_hidden_layers):
                self.bn.append(nn.BatchNorm1d(self.size_hidden_layers, track_running_stats=False))

    def forward(self, x, edge_index):
        z = x
        z = self.encoder(z)

        z = self.in_layer(z, edge_index)
        if self.bn_bool:
            z = self.bn[0](z.permute(0, 2, 1))
        z = self.activation(z.permute(0, 2, 1))

        for n in range(self.nb_hidden_layers - 1):
            z = self.hidden_layers[n](z, edge_index)
            if self.bn_bool:
                z = self.bn[n + 1](z.permute(0, 2, 1))
            z = self.activation(z.permute(0, 2, 1))

        z = self.out_layer(z, edge_index)

        z = self.decoder(z)

        return z


class GCN_NET(torch.nn.Module):
    def __init__(self, hparams, encoder, decoder):
        super(GCN_NET, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.conv1 = nng.GCNConv(hparams['encoder'][-1], hparams['size_hidden_layers'])
        self.conv2 = nng.GCNConv(hparams['size_hidden_layers'], hparams['size_hidden_layers'])
        self.conv3 = nng.GCNConv(hparams['size_hidden_layers'], hparams['size_hidden_layers'])
        self.conv4 = nng.GCNConv(hparams['size_hidden_layers'], hparams['decoder'][0])

    def forward(self, x, edge_index):
        z = self.encoder(x)
        z = z.reshape(z.shape[0], -1, 1)
        z = self.conv1(z, edge_index)
        z = F.gelu(z)
        z = self.conv2(z, edge_index)
        z = F.gelu(z)
        z = self.conv3(z, edge_index)
        z = F.gelu(z)
        z = self.conv4(z, edge_index)
        z = F.gelu(z)
        z = self.decoder(z)
        return z


class GraphSAGEMLP(nn.Module):
    def __init__(self, hparams, encoder, decoder):
        super(GraphSAGEMLP, self).__init__()

        self.nb_hidden_layers = hparams['nb_hidden_layers']
        self.size_hidden_layers = hparams['size_hidden_layers']
        self.bn_bool = hparams['bn_bool']
        self.activation = nn.ReLU()

        self.encoder = encoder
        self.decoder = decoder

        self.in_layer = nng.SAGEConv(
            in_channels=1,
            out_channels=self.size_hidden_layers
        )

        self.hidden_layers = nn.ModuleList()
        for n in range(self.nb_hidden_layers - 1):
            self.hidden_layers.append(nng.SAGEConv(
                in_channels=self.size_hidden_layers,
                out_channels=self.size_hidden_layers
            ))

        self.out_layer = nng.SAGEConv(
            in_channels=self.size_hidden_layers,
            out_channels=hparams['decoder'][0]
        )

        if self.bn_bool:
            self.bn = nn.ModuleList()
            for n in range(self.nb_hidden_layers):
                self.bn.append(nn.BatchNorm1d(self.size_hidden_layers, track_running_stats=False))

    def forward(self, x, edge_index):
        z = self.encoder(x)
        z = z.reshape(z.shape[0], -1, 1)

        z = self.in_layer(z, edge_index)
        if self.bn_bool:
            z = self.bn[0](z.permute(0, 2, 1))
        z = self.activation(z.permute(0, 2, 1))

        for n in range(self.nb_hidden_layers - 1):
            z = self.hidden_layers[n](z, edge_index)
            if self.bn_bool:
                z = self.bn[n + 1](z.permute(0, 2, 1))
            z = self.activation(z.permute(0, 2, 1))

        z = self.out_layer(z, edge_index)

        z = self.decoder(z)

        return z


if __name__ == '__main__':
    import torch
    from model.mlp import MLP
    from torch_geometric.nn import radius_graph

    encoder = MLP(layers=[4, 40000])
    decoder = MLP(layers=[8, 64, 64, 4])
    hparams = {'encoder': [4, 64, 64, 1], 'decoder': [8, 64, 64, 4], 'nb_hidden_layers': 3,
               'size_hidden_layers': 64, 'max_neighbors': 64, 'bn_bool': True,
               'subsampling': 32000, 'r': 0.05}
    # model = GraphSAGEMLP(hparams, encoder, decoder).cuda()
    model = GCN_NET(hparams, encoder, decoder).cuda()
    print(model)

    # data_sampled.pos = data_sampled.pos[idx]
    # data_sampled.x = data_sampled.x[idx]
    # data_sampled.y = data_sampled.y[idx]
    # data_sampled.surf = data_sampled.surf[idx]

    pos = torch.randn(40000, 2).cuda()
    x = torch.randn(8, 4).cuda()
    edge_index = radius_graph(x=pos, r=0.5, loop=True, max_num_neighbors=64)
    y = model(x, edge_index)
    print(y.shape)
