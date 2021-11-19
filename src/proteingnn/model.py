from abc import abstractmethod
from pathlib import Path, PosixPath, WindowsPath
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import torch
import torch.utils.data
import torch.nn as nn
import torch_geometric as torchg
import torch_geometric.nn as gnn
import torch_scatter as torchs
import torchmetrics.functional as mF
from typing import *

from .util import reg_corr_plot, dual_reg_corr_plot, confusion_matrix_plot, dual_confusion_matrix_plot,\
    get_checkpoints

AnyPath = Union[str, PosixPath, WindowsPath]
AnyNum = Union[int, float]


def _concat_step_output(outputs: List[Dict[str, torch.Tensor]], dim: int = 0, skip: Iterable = ('loss',))\
        -> Dict[str, torch.Tensor]:
    """Concatenate step output into single batch tensor.

    Args:
        outputs: Outputs from train/validation/test_step.
        dim: Dimension to be concatenated.
        skip: Keys to be skipped in outputs.
    """
    data = {}
    for output in outputs:
        for key, var in output.items():
            if key in skip:
                continue

            if key in data:
                data[key].append(var)
            else:
                data[key] = [var]

    for key in data:
        data[key] = torch.cat(data[key], dim=dim).clone().detach().cpu()

    return data


def get_default_trainer(
        gpus=None,
        logdir='Log',
        log_name='default',
        restart=True,
        debug=False,
        logging_kwargs: Optional[Dict] = None,
        stop_kwargs: Optional[Dict] = None,
        debug_kwargs: Optional[Dict] = None,
        patience: int = 10,
        min_delta: float = 5e-2,
        max_epochs: int = 1000
    ):
    """Get pytorch_lightning trainer in default configuration.

    Args:
        gpus: GPU device number numbers to be used.
        logdir: Tensorboard root directory.
        log_name: Tensorboard logging directory.
        restart: Restart if a checkpoint file is available under logdir/log_name.
        debug: If true, test run in debug mode.
        logging_kwargs: kwargs for logger.
        stop_kwargs: kwargs for early_stop callback.
        debug_kwargs: kwargs for debugging purpose.
        patience: Patience before early stop (evaluated on validation set).
        min_delta: Minimum difference of validation metric to be considered in patience.
        max_epochs: Max. number of epochs
    """

    checkpoint_kwargs = {
        'default_root_dir': logdir,
        'checkpoint_callback': True,
    }
    if restart:
        ckpt = get_checkpoints(logdir, log_name)[-1]
        if ckpt is not None:
            checkpoint_kwargs['resume_from_checkpoint'] = str(ckpt)

    checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
        monitor='val_mse',
        filename='{epoch:03d}-{val_mse:.3f}',
        save_top_k=5,
        save_last=True,
        mode='min',
        every_n_val_epochs=1,
    )

    logger = pl_loggers.TensorBoardLogger(
        save_dir=logdir,
        name=log_name,
        log_graph=True,
    )
    if logging_kwargs is None:
        logging_kwargs = {
            'weights_summary': 'top',
            'progress_bar_refresh_rate': 1,
            'check_val_every_n_epoch': 1,
            'logger': logger,
        }
    logging_kwargs['logger'] = logger

    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_mse',
        min_delta=min_delta,
        patience=patience,
        verbose=False,
        mode='min',
    )
    if stop_kwargs is None:
        stop_kwargs = {
            'max_epochs': max_epochs,
            'max_time': {'hours': 24},
        }

    run_kwargs = {
        'gpus': gpus,
        'callbacks': [early_stop_callback, checkpoint_callback],
    }

    if debug_kwargs is None:
        debug_kwargs = {
            'fast_dev_run': 2,
            'limit_train_batches': 8,
            'limit_val_batches': 8,
            'profiler': 'simple',
            'deterministic': True,
        }

    if debug:
        return pl.Trainer(**logging_kwargs, **stop_kwargs, **run_kwargs, **checkpoint_kwargs, **debug_kwargs)
    else:
        return pl.Trainer(**logging_kwargs, **stop_kwargs, **run_kwargs, **checkpoint_kwargs)


def gnn_module(in_channels: int, hidden_channels: int, n_layers: int, leakyrelu_slope: float,
               dropout: float = 0., gnn_class: Any = 'GCNConv', gnorm_class: Any = None,
               gnn_args: str = 'x, edge_index -> x', **kwargs):
    """Stremalined GNN submodule.

    Does not support edge convolution.

    Args:
        in_channels: Input channel size.
        hidden_channels: Hidden channel size.
        n_layers: Number of layers.
        leakyrelu_slope: LeakyReLU slope.
        dropout: Dropout at the final layer.
        gnn_class: Conv name in string or class in pytorch_geometric.
        gnorm_class: Normalizaton name in string or class in pytorch_geometric.
        gnn_args: Argument input type for gnn_class in pytorch_geometric.nn.Sequential format.
        kwargs: kwargs for gnn_class.
    """
    assert n_layers >= 1

    if type(gnn_class) == str:
        gnn_class = getattr(gnn, gnn_class)
    elif gnorm_class is None:
        pass
    elif not issubclass(gnn_class, gnn.MessagePassing):
        raise TypeError(f'{gnn_class} is neither str nor MessagePassing subclass.')

    if type(gnorm_class) == str and gnorm_class:
        gnorm_class = getattr(gnn, gnorm_class)
    # elif gnorm_class is None:
    #     pass
    # elif not issubclass(gnorm_class, gnn.MessageNorm):  # BaseClass unavailable
    #     raise TypeError(f'{gnorm_class} is neither str nor MessageNorm subclass.')

    layers = [
        (gnn_class(in_channels, hidden_channels, **kwargs), gnn_args),
        nn.LeakyReLU(leakyrelu_slope)
    ]
    if gnorm_class:  # gnorm_class instantiation argument not supported
        layers.append(
            (gnorm_class(hidden_channels), 'x, batch -> x')
        )

    for _ in range(n_layers-1):
        layers.extend([
            (gnn_class(hidden_channels, hidden_channels, **kwargs), gnn_args),
            nn.LeakyReLU(leakyrelu_slope)
        ])
        if gnorm_class:
            layers.append(
                (gnorm_class(hidden_channels), 'x, batch -> x')
            )

    if dropout:
        layers.append(nn.Dropout(p=dropout))

    model = gnn.Sequential('x, edge_index, batch', layers)
    return model


def mlp_module(in_channels: int, hidden_channels: int, n_layers: int, leakyrelu_slope: float,
               dropout: float, norm_name: Optional[str] = None, norm_final_layer: bool = True):
    """Stremalined MLP submodule.

    Args:
        in_channels: Input channel size.
        hidden_channels: Hidden channel size.
        n_layers: Number of layers.
        leakyrelu_slope: LeakyReLU slope.
        dropout: Dropout at the final layer.
        norm_name: Normalization name in PyTorch.nn.
        norm_final_layer: If true, only do normalization at the final layer.
    """

    # norm_name only support type str
    assert n_layers >= 1
    layers = [
        nn.Linear(in_channels, hidden_channels),
        nn.LeakyReLU(leakyrelu_slope)
    ]
    if norm_name:
        layers.append(getattr(nn, norm_name)(hidden_channels))

    for i in range(n_layers-1):
        layers.extend([
            nn.Linear(hidden_channels, hidden_channels),
            nn.LeakyReLU(leakyrelu_slope)
        ])
        if norm_name and (i != n_layers - 2 or norm_final_layer):
            layers.append(getattr(nn, norm_name)(hidden_channels))

    if dropout:
        layers.append(nn.Dropout(p=dropout))

    model = nn.Sequential(*layers)
    return model


class BaseModel(pl.LightningModule):
    """Abstract base class for either regression or classification.

    (warning: classification is not tested.)
    """

    def __init__(self, criterion: Optional[Callable] = None, regression: bool = True, num_classes: Optional[int] = None,
                 lr: float = 1e-3, betas: Optional[Tuple] = None, weight_decay: float = 0.,
                 alphal2: Optional[Dict] = None, log_train: bool = True, log_plot: bool = True):
        """Initialize with hyperparameters.

        Args:
            criterion: Loss function between prediction and target.
            regression: Indicate regression or classification task.
            num_classes: If not regression, the number of class in classification.
            lr: Learning rate in Adam optimizer.
            betas: Beta-s in Adam optimizer.
            weight_decay: L2 regularization strength.
            alphal2: Dictionary for layer specific L2 regularization. Key indicates layer name. Value indicates
                regularization strength.
            regression: If true, train and log in regression mode. Otherwise classification mode.
            log_train: Log training in tensorboard.
            log_plot: Plot logging performance in tensorboard.
        """
        super().__init__()
        self.model = None
        self._regression = regression
        self.final_layer = self._set_final_layer()

        if self._regression:
            self.num_classes = None
        elif not self.num_classes:
            raise ValueError(f'Invalid num_classes {num_classes} for classification.')
        else:
            self.num_classes = num_classes

        if criterion:
            self.loss = criterion
        else:
            self.loss = nn.MSELoss() if self._regression else nn.BCEWithLogitsLoss()

        self.alphal2 = alphal2.copy() if alphal2 else {}
        self.betas = betas if betas else (0.9, 0.999)

        self.log_train = log_train
        self.log_plot = log_plot
        self.dual_dict = {'train': None, 'val': None}

    def _l2_regularizer(self):
        """Get regularization cost."""
        if not self.alphal2:
            return 0

        l2 = 0
        for name, parameter in self.named_parameters():
            for alpha_name, alpha_value in self.alphal2.items():
                if alpha_name in name:
                    l2 += alpha_value * (parameter ** 2).sum()

            if name not in self.alphal2 and 'all' in self.alphal2:
                l2 += self.alphal2['all'] * (parameter ** 2).sum()

        return l2

    def configure_optimizers(self):
        """Initialize optimizer"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay,
                                     betas=self.betas)
        return optimizer

    # def forward(self, x, *args, **kwargs):
    #     raise NotImplementedError

    @abstractmethod
    def _step(self, *args, **kwargs) -> Dict[str, Any]:
        """Abstract method for step."""
        raise NotImplementedError

    def training_step(self, data, batch_idx):
        """Training step."""
        return self._step(data, batch_idx, stage='train')

    def validation_step(self, data, batch_idx):
        """Validation step."""
        return self._step(data, batch_idx, stage='val')

    def training_epoch_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        """Aggregate training step logging information and plot."""
        if self.global_step == 0:
            self.dual_dict['train'] = None
            return

        if self.log_train:
            outputs = _concat_step_output(outputs)
            if self._regression:
                self._log_scaler_reg(outputs['y'], outputs['pred'], stage='train')
            else:
                self._log_scaler_cls(outputs['y'], outputs['pred'], stage='train', **kwargs)
                self._log_pr_curve(outputs['y'], outputs['pred'])

            self.dual_dict['train'] = outputs  # dual plot in validation_epoch_end

    def validation_epoch_end(self, outputs: Dict[str, Any]) -> None:
        """Aggregate validation step logging information and plot."""
        if self.global_step == 0:
            return

        outputs = _concat_step_output(outputs)
        if self._regression:
            self._log_scaler_reg(outputs['y'], outputs['pred'], stage='val')
        else:
            self._log_scaler_cls(outputs['y'], outputs['pred'], stage='val')

        if self.log_plot:  # overlay train and val visualization together
            self._log_plot_reg(outputs) if self._regression else self._log_plot_cls(outputs)

    def _set_final_layer(self):
        """Set final layer depending on regression or classification task."""
        if self._regression:
            return nn.LazyLinear(out_features=1)
        else:
            return nn.LazyLinear(out_features=self.num_classes)

    def _log_scaler_reg(self, y: torch.Tensor, pred: torch.Tensor, stage: str):
        """Log regression performance."""
        assert self._regression
        metrics = {
            f'{stage}_mse': mF.mean_squared_error(pred, y),
            f'{stage}_pcc': mF.pearson_corrcoef(pred, y),
            f'{stage}_src': mF.spearman_corrcoef(pred, y),
            f'{stage}_r2': mF.r2score(pred, y),
        }
        metrics[f'{stage}_mse'] = metrics[f'{stage}_mse'].to(self.device)  # early stop monitor requires same device
        self.log_dict(metrics, on_epoch=True, logger=True)

    def _log_scaler_cls(self, y: torch.Tensor, pred: torch.Tensor, stage: str, average: str = 'micro'):
        """Log classification performance."""
        assert not self._regression
        common_kwargs = {'average': average, 'num_classes': self.num_classes}

        metrics = {
            f'{stage}_accuracy': mF.accuracy(pred, y, **common_kwargs),
            f'{stage}_auroc': mF.auroc(pred, y, **common_kwargs),
            f'{stage}_f1': mF.f1(pred, y, **common_kwargs),
            f'{stage}_mcc': mF.matthews_corrcoef(pred, y, num_classes=self.num_classes),
        }
        metrics[f'{stage}_mse'] = metrics[f'{stage}_mse'].to(self.device)  # early stop monitor requires same device
        self.log_dict(metrics, on_epoch=True, logger=True)

    def _log_plot_reg(self, outputs: Dict[str, Any]):
        """Plot regression log."""
        assert self._regression
        if self.log_train and self.dual_dict['train'] is not None:  # None if restart
            self.dual_dict['val'] = outputs
            fig = dual_reg_corr_plot(self.dual_dict)
        else:
            fig = reg_corr_plot(outputs['y'], outputs['pred'])

        tensorboard = self.logger.experiment
        tensorboard.add_figure(f'regression_corr', fig, global_step=self.global_step)

    def _log_plot_cls(self, outputs: Dict[str, Any]):
        """Plot classification log."""
        assert not self._regression
        tensorboard = self.logger.experiment
        cm = self._get_cm(outputs['y'], outputs['prediction'])

        if self.log_train and self.dual_dict['train'] is not None:
            self.dual_dict['val'] = outputs
            cms = [cm, self._get_cm(self.dual_dict['train']['y'], self.dual_dict['train']['prediction'])]
            fig = dual_confusion_matrix_plot(cms)
        else:
            fig = confusion_matrix_plot(self._get_cm(outputs['prediction'], outputs['y']))

        tensorboard.add_figure(f'confusion_matrix', fig, global_step=self.global_step)
        self._log_pr_curve(outputs['y'], outputs['prediction'])

    def _log_pr_curve(self, y, pred):
        """Log PR curve in classification."""
        assert not self._regression
        tensorboard = self.logger.experiment
        for class_i in range(self.num_classes):
            tensorboard.add_pr_curve(f'PR_curve_class_{class_i}', pred[:, class_i], y[:, class_i],
                                     global_step=self.global_step)

    def _get_cm(self, y, pred):
        """Get confusion matrix in classification."""
        assert not self._regression
        cm = mF.confusion_matrix(pred, y, num_classes=self.num_classes)
        return cm.clone().detach().cpu().numpy()


class BaseGNNModel(BaseModel):
    """Abstract GNN base class."""
    def _step(self, data, batch_idx, stage):
        """Step for GNN."""
        pred = self.forward(data.x, data.edge_index, data.batch)
        if len(pred.shape) > 1:
            pred = pred.squeeze(1)

        loss = self.loss(pred, data.y) + self._l2_regularizer()
        self.log(f'{stage}_loss', loss, on_epoch=True, logger=True)
        return {'loss': loss, 'y': data.y, 'pred': pred}

    def forward(self, x, edge_index, batch):
        """Forward for GNN."""
        if self.model is None:
            raise NotImplementedError('Either no model set or forward method not overridden.')
        else:
            x = self.model.forward(x, edge_index, batch)
            return self.final_layer(x)


class BaseMLPModel(BaseModel):
    """Abstract MLP base class which takes graph input"""
    def __init__(self, num_nodes: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_nodes = num_nodes

    def _step(self, data, batch_idx, stage):
        pred = self.forward(data.x)
        if len(pred.shape) > 1:
            pred = pred.squeeze(1)

        loss = self.loss(pred, data.y) + self._l2_regularizer()
        self.log(f'{stage}_loss', loss, on_epoch=True, logger=True)
        return {'loss': loss, 'y': data.y, 'pred': pred}

    def forward(self, x) -> torch.Tensor:
        # re-batch to non-graph input
        batch_size = x.shape[0] // self.num_nodes
        x = x.reshape((batch_size, x.shape[0] * x.shape[1] // batch_size))

        if self.model is None:
            raise NotImplementedError('Either no model set or forward method not overridden.')
        else:
            x = self.model.forward(x)
            return self.final_layer(x)


class global_softmax_pooling(torch.nn.Module):
    """Global softmax aggregation.

    Reference:
    Li, Guohao, et al. "Deepergcn: All you need to train deeper gcns." arXiv preprint arXiv:2006.07739 (2020).
    """

    def __init__(self, reduce: str = 'mean', beta: float = 1.):
        """Initialize with hyperparameter.

        Args:
            reduce: Aggregation method, by default mean. Available options are mean, max, sum.
            beta: Amplification factor in softmax.
        """
        super().__init__()
        self.reduce = reduce
        self.beta = beta

    def forward(self, x, batch, size=None):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = batch[-1].item() + 1 if size is None else size

        x_ = torchg.utils.softmax(self.beta * x, batch, num_nodes=size)
        out = torchs.scatter(x * x_, batch, dim=0, reduce=self.reduce)
        return out

    def __repr__(self):
        return f'{self.__class__.__name__} ({self.reduce})'


class global_powermean_pooling(torch.nn.Module):
    """Global power-mean

    Reference:
    Li, Guohao, et al. "Deepergcn: All you need to train deeper gcns." arXiv preprint arXiv:2006.07739 (2020).
    """

    def __init__(self, p: float = 1):
        """ Initialize

        Args:
            p: Power in power-mean. By default, p=1, which means arithmatic mean. Range of p is (-inf, inf).
        """
        super().__init__()
        self.p = p

    def forward(self, x, batch):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        x = torchs.scatter_mean(x ** self.p, batch, dim=0)
        return x ** (1/self.p)

    def __repr__(self):
        return f'{self.__class__.__name__} (p={self.p})'


class softmax_pooling(nn.Module):
    """Softmax pooling for tensor-like (non-graph) data."""
    def __init__(self, beta: float = 1., dim: int = 0):
        super().__init__()
        self.beta = beta
        self.dim = dim

    def forward(self, x):
        weights = [torch.nn.Softmax(dim=self.dim)(self.beta * x_) for x_ in x]
        weights = torch.stack(weights)
        return torch.sum(x * weights, dim=1)


class ResGraphConv(gnn.MessagePassing):
    """(Legacy) Residual Graph Conv."""
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels)
        self.gcn = gnn.GCNConv(in_channels, out_channels, **kwargs)

    def forward(self, x, edge_index):
        return self.lin(x) + self.gcn(x, edge_index)

    def __repr__(self):
        return 'ResGraphConv'


class DummyGraphConv(gnn.MessagePassing):
    """A graph conv ignoring edges.

    Essentially linear transformation on node features."""
    def __init__(self, in_channels: Optional[int] = None, out_channels: Optional[int] = None, identity: bool = False):
        """Initialize.

        Args:
            in_channels: Input channel size. Skipped if identity option is True.
            out_channels: Output_channel size Skipped if identity option is True.
            identity: If true, use identity mapping in no-edge GNN. Otherwise use nn.Linear(in_channels, out_channels).
        """
        super().__init__()
        self.lin = nn.Identity() if identity else nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.lin(x)

    def __repr__(self) -> str:
        return 'DummyGraphConv'

