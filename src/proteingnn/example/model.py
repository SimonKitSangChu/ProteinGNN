from ..model import *


class ConcatGCNModel(BaseGNNModel):
    """Mimic of DeepFRI architecture which has connection directly from sequence to prediction.

    Reference:
    Gligorijević, Vladimir, et al. "Structure-based protein function prediction using graph convolutional networks."
    Nature communications 12.1 (2021): 1-14.
    """
    def __init__(self,
                 in_channels: int = 1280,
                 hidden_channels: int = 16,
                 gnn_class: Any = 'GCNConv',
                 gnorm_name: Any = None,
                 leakyrelu_slope: float = 0.,
                 beta: float = 1.,
                 lr: float = 1e-3,
                 weight_decay: float = 2e-4,
                 dropout: float = 0.3,
                 regression: bool = True,
                 **model_kwargs
                 ):
        super().__init__(regression=regression, weight_decay=weight_decay, lr=lr, betas=None)
        self.save_hyperparameters()
        self.save_hyperparameters(*model_kwargs)

        if gnn_class == 'GCNConv' and not model_kwargs:  # Reference to DeepFRI architecture
            model_kwargs = {'bias': False}

        common_kwargs = {
            'n_layers': 1,
            'leakyrelu_slope': leakyrelu_slope,
            'gnn_class': gnn_class,
            'gnorm_name': gnorm_name,
            **model_kwargs
        }
        self.glayer1 = gnn_module(in_channels=in_channels, hidden_channels=hidden_channels, **common_kwargs)
        self.glayer2 = gnn_module(in_channels=hidden_channels, hidden_channels=hidden_channels, **common_kwargs)
        self.glayer3 = gnn_module(in_channels=hidden_channels, hidden_channels=hidden_channels, **common_kwargs)
        self.gsp = global_softmax_pooling(beta=beta)
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 3, hidden_channels),
            nn.LeakyReLU(leakyrelu_slope)
        )

    def forward(self, x, edge_index, batch):
        x1 = self.glayer1.forward(x, edge_index, batch)
        x2 = self.glayer2.forward(x1, edge_index, batch)
        x3 = self.glayer3.forward(x2, edge_index, batch)
        x4 = torch.cat([x1, x2, x3], dim=1)  # combine into gnn.Sequential
        x5 = self.gsp.forward(x4, batch)
        x6 = self.mlp.forward(x5)
        return self.final_layer.forward(x6)

    def _get_layer(self, conv_name, in_channels, hidden_channels, gnorm_name, leakyrelu_slope, **kwargs):
        """Utility function to generate graph layer."""
        layer = [
            (getattr(gnn, conv_name)(in_channels, hidden_channels, **kwargs), 'x, edge_index -> x'),
            nn.LeakyReLU(leakyrelu_slope),
        ]
        if gnorm_name:
            layer.append(getattr(gnn, gnorm_name))

        return gnn.Sequential('x, edge_index, batch', layer)


class FastGCNModel(BaseGNNModel):
    """Streamlined graph neural network.

    Only support inverse temperature of 1 in graph pooling. Does not support edge convolution or update.
    """
    def __init__(self,
                 gnn_class: Any,
                 in_channels: int,
                 hidden_channels: int,
                 n_gnn_layers: int = 3,
                 n_lin_layers: int = 1,
                 dropout_lin: float = 0.0,
                 lr: float = 1e-3,
                 betas: Optional[Tuple] = None,
                 weight_decay: float = 0.,
                 norm_name: Optional[str] = None,
                 gnorm_class: Any = None,
                 leakyrelu_slope: float = 0.2,
                 alphal2: Optional[Dict] = None,
                 regression: bool = True,
                 batch_size: int = 1,  # for backward compatibility
                 **model_kwargs):
        """Initialize with hyperparameters.

        Args:
            gnn_class: Conv name in string or class in pytorch_geometric.
            in_channels: Input channel size
            hidden_channels: Hidden channel size. The same throughout GNN and MLP layers.
            n_gnn_layers: Number of GNN layers.
            n_lin_layers: Number of MLP layers.
            dropout_lin: Dropout in MLP layer.
            lr: Learning rate in Adam optimizer.
            betas: Beta-s in Adam optimizer.
            weight_decay: L2 regularization strength.
            norm_name: Normalzaition layer name in PyTorch.
            gnorm_class: Normalization layer name or class in pytorch_geometric.
            leakyrelu_slope: Slope in LeakyReLU.
            alphal2: Dictionary for layer specific L2 regularization. Key indicates layer name. Value indicates
                regularization strength.
            regression: If true, train and log in regression mode. Otherwise classification mode.
            batch_size: Batch size logging for backward compatibility and hyperparameter logging.
        """

        super().__init__(regression=regression, lr=lr, betas=betas, weight_decay=weight_decay, alphal2=alphal2)
        self.save_hyperparameters()
        self.save_hyperparameters(*model_kwargs)

        # pre-define in-house class from string
        if gnn_class == 'ResGraphConv':
            gnn_class = ResGraphConv
        elif gnn_class == 'DummyGraphConv':
            gnn_class = DummyGraphConv

        gnn_layer = gnn_module(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            n_layers=n_gnn_layers,
            leakyrelu_slope=leakyrelu_slope,
            gnn_class=gnn_class,
            gnorm_class=gnorm_class,
            **model_kwargs
        )
        mlp_layer = mlp_module(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            n_layers=n_lin_layers,
            leakyrelu_slope=leakyrelu_slope,
            dropout=dropout_lin,
            norm_name=norm_name
        )
        self.model = gnn.Sequential('x, edge_index, batch', [
            (gnn_layer, 'x, edge_index, batch -> x'),
            (global_softmax_pooling(reduce='mean'), 'x, batch -> x'),
            mlp_layer,
        ])


class FastMLPModel(BaseMLPModel):
    """Streamlined multilayer perceptron which takes graph input.

    Automatically rebatch graph sample for MLP. Assume consistent node orders.
    """
    def __init__(self,
                 num_nodes: int,
                 n_lin_layers: int,
                 in_channels: int,
                 hidden_channels: int = 16,
                 dropout: float = 0.,
                 leakyrelu_slope: float = 0.2,
                 lr: float = 1e-3,
                 betas: Optional[Tuple] = None,
                 weight_decay: float = 0,
                 alphal2: Optional[Dict] = None,
                 norm_name: Optional[str] = 'BatchNorm1d',
                 regression: bool = True,
                 batch_size: int = 1,  # for backward compatibility
                 ):
        """Initialize with hyperparameters.

        Args:
            num_nodes: Number of nodes in graph. Must be fixed in all samples.
            n_lin_layers: Number of MLP layers.
            in_channels: Input channel size
            hidden_channels: Hidden channel size. The same throughout GNN and MLP layers.
            n_lin_layers: Number of MLP layers.
            lr: Learning rate in Adam optimizer.
            betas: Beta-s in Adam optimizer.
            weight_decay: L2 regularization strength.
            norm_name: Normalzaition layer name in PyTorch.
            leakyrelu_slope: Slope in LeakyReLU.
            alphal2: Dictionary for layer specific L2 regularization. Key indicates layer name. Value indicates
                regularization strength.
            regression: If true, train and log in regression mode. Otherwise classification mode.
            batch_size: Batch size logging for backward compatibility and hyperparameter logging.
        """
        super().__init__(regression=regression, lr=lr, betas=betas, weight_decay=weight_decay, alphal2=alphal2,
                         num_nodes=num_nodes)
        self.save_hyperparameters()

        self.model = mlp_module(
            in_channels=num_nodes * in_channels,
            hidden_channels=hidden_channels,
            n_layers=n_lin_layers,
            leakyrelu_slope=leakyrelu_slope,
            dropout=dropout,
            norm_name=norm_name
        )


class SingleSiteMLP(BaseMLPModel):
    """Simple MLP model for mutational embedding at only mutation site."""
    def __init__(self,
                 n_lin_layers: int,
                 in_channels: int,
                 hidden_channels: int = 16,
                 dropout: float = 0.,
                 leakyrelu_slope: float = 0.2,
                 lr: float = 1e-3,
                 betas: Optional[Tuple] = None,
                 weight_decay: float = 0,
                 alphal2: Optional[Dict] = None,
                 norm_name: Optional[str] = 'BatchNorm1d',
                 regression: bool = True,
                 ):
        """Initialize with hyperparameters.

        Args:
            n_lin_layers: Number of MLP layers.
            in_channels: Input channel size
            hidden_channels: Hidden channel size. The same throughout GNN and MLP layers.
            lr: Learning rate in Adam optimizer.
            betas: Beta-s in Adam optimizer.
            weight_decay: L2 regularization strength.
            norm_name: Normalzaition layer name in PyTorch.
            leakyrelu_slope: Slope in LeakyReLU.
            alphal2: Dictionary for layer specific L2 regularization. Key indicates layer name. Value indicates
                regularization strength.
            regression: If true, train and log in regression mode. Otherwise classification mode.
        """
        super().__init__(regression=regression, lr=lr, betas=betas, weight_decay=weight_decay, alphal2=alphal2,
                         num_nodes=1)
        self.save_hyperparameters()

        self.model = mlp_module(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            n_layers=n_lin_layers,
            leakyrelu_slope=leakyrelu_slope,
            dropout=dropout,
            norm_name=norm_name
        )

    def _step(self, data, batch_idx, stage):
        x, y = data
        pred = self.forward(x)
        if len(pred.shape) > 1:
            pred = pred.squeeze(1)

        loss = self.loss(pred, y) + self._l2_regularizer()
        self.log(f'{stage}_loss', loss, on_epoch=True, logger=True)
        return {'loss': loss, 'y': y, 'pred': pred}

    def forward(self, x) -> torch.Tensor:
        x1 = self.model.forward(x)
        return self.final_layer(x1)


class SeqPoolingMLP(BaseGNNModel):
    """Simple MLP model after global pooling."""
    def __init__(self,
                 n_layers: int = 3,
                 in_channels: int = 1280,
                 hidden_channels: int = 16,
                 norm_name: Optional[str] = 'BatchNorm1d',
                 dropout: float = 0.,
                 leakyrelu_slope: float = 0.2,
                 lr: float = 1e-3,
                 weight_decay: float = 0,
                 alphal2: Optional[Dict] = None,
                 beta: float = 1.,
                 regression: bool = True,
                 ):
        """Initialize with hyperparameters.

        Args:
            n_layers: Number of MLP layers.
            in_channels: Input channel size
            hidden_channels: Hidden channel size. The same throughout GNN and MLP layers.
            lr: Learning rate in Adam optimizer.
            weight_decay: L2 regularization strength.
            norm_name: Normalzaition layer name in PyTorch.
            leakyrelu_slope: Slope in LeakyReLU.
            alphal2: Dictionary for layer specific L2 regularization. Key indicates layer name. Value indicates
                regularization strength.
            beta: Inverse temperature in global softmax aggregation.
            regression: If true, train and log in regression mode. Otherwise classification mode.
        """
        super().__init__(regression=regression, lr=lr,  weight_decay=weight_decay, alphal2=alphal2)
        self.save_hyperparameters()

        pre_layer = gnn_module(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            n_layers=1,
            leakyrelu_slope=leakyrelu_slope,
            gnn_class=DummyGraphConv,
            gnorm_class=None,
        )
        post_layer = mlp_module(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            leakyrelu_slope=leakyrelu_slope,
            dropout=dropout,
            norm_name=norm_name,
        )
        self.model = gnn.Sequential('x, edge_index, batch', [
            (pre_layer, 'x, edge_index, batch -> x'),
            (global_softmax_pooling(reduce='mean', beta=beta), 'x, batch -> x'),
            post_layer
        ])
