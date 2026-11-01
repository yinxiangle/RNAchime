

import torch
import torch.nn as nn
from dgl.nn.pytorch.glob import SumPooling,GlobalAttentionPooling, MaxPooling
from dgl.nn.pytorch.conv import HGTConv
import pickle
from tqdm import tqdm
import numpy as np
from .learn import send_graph_to_device

class Attributor(nn.Module):
    """
    A module for attributing features to a given input.
    """
    def __init__(self, dims, dropout=0.5):
        super(Attributor, self).__init__()
        self.dims = dims
        self.dropout = dropout
        self.build_model()

    def build_model(self):
        layers = []

        short = self.dims[:-1]
        last_hidden, last = (*self.dims[-2:],)

        for dim_in, dim_out in zip(short, short[1:]):
            layers.append(nn.Linear(dim_in, dim_out))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
        layers.append(nn.Linear(last_hidden, last))
      

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class Embedder(nn.Module):
    """
        Model for producing node embeddings.
    """
    def __init__(self, head_size, num_heads, dropout=0.2, num_etypes=13, num_ntypes=5):
        super(Embedder, self).__init__()
        self.head_size = head_size
        self.num_heads = num_heads
        self.num_etypes = num_etypes
        self.num_ntypes = num_ntypes
        self.dropout = dropout
        self.layers = self.build_model()

    def build_model(self):
        if len(self.head_size)!=len(self.num_heads):
            print("length of head_size and num_heads should be same")
            exit()
        layers = nn.ModuleList()

        # input feature is just node degree
        i2h = self.build_layer(1, self.head_size[0], self.num_heads[0])
        layers.append(i2h)


        for hidden_layer_dim in range(len(self.head_size)-1):
            h2h = self.build_layer(self.head_size[hidden_layer_dim]*self.num_heads[hidden_layer_dim], self.head_size[hidden_layer_dim+1],
                                          self.num_heads[hidden_layer_dim+1])
            layers.append(h2h)
        return layers

    @property
    def current_device(self):
        """
        :return: current device this model is on
        """
        return next(self.parameters()).device

    def build_layer(self, in_dim, head_size, num_heads):
        return HGTConv(in_dim, head_size, num_heads, self.num_ntypes, self.num_etypes, dropout=self.dropout)
    
    def forward(self, g):
        h = g.in_degrees().view(-1, 1).float().to(self.current_device)
        for layer in self.layers:
            h = layer(g, h, g.ndata['one_hot'], g.edata['one_hot'])
        g.ndata['h'] = h
        del h
        embeddings = g.ndata.pop('h')
        return embeddings


class Model(nn.Module):
    def __init__(self, head_size, num_heads, device, criterion, attributor_dims, num_ntypes, num_etypes, pool='max',
                dropout=0.2):
        """

        :param head_size: the out put embeddings dimensions
        :param attributor_dims: the number of motifs to look for
        :param num_etypes: the number of possible edge types
        :param num_ntypes: technical rGCN option
        :param rec: the constant in front of reconstruction loss
        :param mot: the constant in front of motif detection loss
        :param orth: the constant in front of dictionnary orthogonality loss
        :param attribute: Wether we want the network to use the attribution module
        """
        super(Model, self).__init__()
        self.head_size = head_size
        self.criterion = criterion
        self.register_buffer('fp_mean', torch.zeros(attributor_dims[-1]))
        self.register_buffer('fp_std', torch.ones(attributor_dims[-1]))
        self.attributor_dims = attributor_dims
        self.num_etypes = num_etypes
        self.num_ntypes = num_ntypes
        self.num_heads=num_heads
        self.device = device
        self.dropout = dropout
        
        if pool == 'att':
            pooling_gate_nn = nn.Linear(attributor_dims[0], 1)
            self.pool = GlobalAttentionPooling(pooling_gate_nn)
        elif pool == 'max':
            self.pool = MaxPooling()
        else:
            self.pool = SumPooling()

        self.embedder = Embedder(head_size=self.head_size, num_heads=self.num_heads,
                                 num_etypes=self.num_etypes, num_ntypes=self.num_ntypes,
                                 dropout=self.dropout)

        self.attributor = Attributor(attributor_dims, dropout=self.dropout)

    def forward(self, g):
        embeddings = self.embedder(g)
        fp = self.pool(g, embeddings)
        fp = self.attributor(fp)
        return fp, embeddings
    
    def set_fp_stats(self, mean, std):
        self.fp_mean = mean
        self.fp_std = std

    def _normalize_fp(self, x, eps=1e-6):
        return (x - self.fp_mean) / (self.fp_std + eps)
    def fp_loss(self, target_fp, pred_fp):
        if isinstance(self.criterion, nn.HuberLoss) and self.attributor_dims[-1] == 512:
            target_fp = self._normalize_fp(target_fp)
            
        loss = self.criterion(pred_fp, target_fp)
        return loss

def load_model(run,name,device='cpu'):
    """
        Load full trained model with id `run`

    """

    meta = pickle.load(open(f'{run}/meta.p', 'rb'))

    edge_map = meta['edge_map']
    num_edge_types = len(edge_map)

    model_dict = torch.load(f'{run}/{name}.pth', map_location='cpu')
    criterion_name = meta.get('criterion', 'HuberLoss').lower()
    if criterion_name in {'bcewithlogitsloss', 'bcewithlogits', 'bceloss', 'bce'}:
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(meta.get('pos_weight', 1.0)))
    elif criterion_name in {'huberloss', 'huber'}:
        criterion = torch.nn.HuberLoss(delta=meta.get('delta', 1.0))
    else:
        raise ValueError(f'Unsupported criterion in checkpoint metadata: {criterion_name}')
    model = Model(head_size=meta['head_size'],
                  num_heads=meta['num_heads'],
                  attributor_dims=meta['attributor_dims'],
                  criterion=criterion,
                  num_ntypes=5,
                  num_etypes=num_edge_types,
                  device=device,
                  pool=meta['pool'],
                  dropout=meta['dropout'])
    model.load_state_dict(model_dict['model_state_dict'],strict=False)
    return model, meta


def predict(model, loader, max_graphs=10, device='cpu'):
    all_graphs = loader.dataset.all_graphs
    Z = []
    fps = []
    g_inds = []

    model = model.to(device)
    with torch.no_grad():
        model.eval()
        for i, (graph, fp, graph_index) in tqdm(enumerate(loader), total=len(loader)):

            graph = send_graph_to_device(graph, device)
            fp, z = model(graph)
            g_inds.extend(graph_index.tolist())
            Z.append(z.cpu().numpy())
            fps.append(fp.cpu().numpy())
            
    Z = np.concatenate(Z)
    fps = np.concatenate(fps)
    g_inds = np.asarray(g_inds)
    return fps, Z , g_inds , all_graphs