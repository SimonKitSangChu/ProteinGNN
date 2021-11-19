from Bio import PDB
import collections
from pathlib import Path, PosixPath, WindowsPath
import pandas as pd
import numpy as np
from matplotlib.figure import Figure, Axes
import matplotlib.colors
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import ConfusionMatrixDisplay, r2_score
import scipy
import torch
import torch_geometric as torchg
from typing import *

plt.style.use('ggplot')
AnyPath = Union[str, PosixPath, WindowsPath]
CorrType = Dict[str, Tuple[float, float]]
OrderedIterableType = Union[List, OrderedDict, Tuple]


def get_checkpoint_from_version(version_dir: AnyPath, metric_name: Optional[str] = None, sort_order: str = 'min',
                                return_all: bool = False) -> Union[List[AnyPath], AnyPath]:
    """Get checkpoint from version directory in tensorboard format

    Args:
        version_dir: Path to version directory.
        metric_name: Name of the best metric. Requires checkpoint name format as -
            {anything}{metric_name}={metric_value}.ckpt
        sort_order: Sort order to choose best metric from. Apply only when metric_name is specified.
            Available options - 'min' or 'max'.
        return_all: If true, return all checkpoints.
    """
    ckpt_dir = Path(version_dir) / 'checkpoints'
    if not ckpt_dir.exists():
        raise FileNotFoundError(f'{ckpt_dir} not found.')

    if metric_name:
        ckpt_dict = {}
        for ckpt in ckpt_dir.glob(f'*{metric_name}*.ckpt'):
            ckpt_name = ckpt.stem
            metric_value = float(ckpt_name.split(f'{metric_name}=')[-1])
            ckpt_dict[ckpt_name] = metric_value

        if ckpt_dict:
            ckpt_names = sorted(ckpt_dict, key=ckpt_dict.get, reverse=sort_order != 'min')
            ckpts = [ckpt_dir / f'{ckpt_name}.ckpt' for ckpt_name in ckpt_names]
            if return_all:
                return ckpts
            else:
                return ckpts[0]
        else:
            raise FileNotFoundError(f'No checkpoint file found for logged metric name {metric_name}'
                                    f'under {ckpt_dir}.')

    if return_all:
        ckpts = ckpt_dir.glob(f'*{metric_name}*.ckpt')
        if not ckpts:
            return ckpts
        else:
            raise FileNotFoundError(f'No checkpoint file found under {ckpt_dir}.')

    ckpt = ckpt_dir / 'last.ckpt'
    if not ckpt.exists():
        raise FileNotFoundError(f'Last.ckpt not found under {ckpt_dir}.')

    return ckpt


def get_version_dirs(rootdir: AnyPath = 'Log', log_name: Optional[str] = 'default', deep: bool = True) -> List[AnyPath]:
    """Get all version directories under rootdir.

    If not deep, assumes version directory has a name of version_{i}. Assume tensorboard logging hierarchy.
    rootdir - log directory - version directory - checkpoint directory

    Args:
        rootdir: Version directory to look underneath from.
        log_name: Name of log directory.
        deep: Search for version directory in depth for non-standard directory hierarchy.
    """

    if log_name:
        logdir = Path(f'{rootdir}/{log_name}')
    else:
        logdir = Path(rootdir)

    version_dirs = [pth for pth in logdir.glob('version_*') if pth.is_dir()]
    if deep:
        deep_version_dirs = [pth for pth in logdir.glob('**/version_*') if pth.is_dir()]
        version_dirs.extend(deep_version_dirs)

    if not version_dirs:
        raise FileNotFoundError(f'No version directory under {logdir}.')

    return version_dirs


def get_checkpoints(logdir: AnyPath = 'Log', log_name: Optional[str] = 'default', deep: bool = True,
                    metric_name: Optional[str] = None, sort_order: str = 'min') -> Union[PosixPath, WindowsPath, None]:
    """Retrieve checkpoint file path. Return None if not found.

    args:
        version: Version to restart from. By default last version.
        metric_name: Name of the best metric. Requires checkpoint name format as -
            {anything}{metric_name}={metric_value}.ckpt
        sort_order: Sort order to choose best metric from. Apply only when metric_name is specified.
            Available options - 'min' or 'max'.
    """
    ckpts = {}
    for version_dir in get_version_dirs(rootdir=logdir, log_name=log_name, deep=deep):
        ckpt = get_checkpoint_from_version(version_dir, metric_name=metric_name, sort_order=sort_order)
        ckpts[version_dir.stem] = ckpt

    ckpts = [ckpts[key] for key in sorted(ckpts)]
    return ckpts


def evaluate_versions(class_: Any, dataset: Any, ckpt: AnyPath, is_graph: bool = True, logdir: AnyPath = 'Log', log_name: Optional[str] = 'default',
                      deep: bool = True, metric_name: Optional[str] = None, sort_order: str = 'min')\
        -> List[Dict[str, float]]:
    """Evaluate all checkpoints on x, y.

    Args:
        x: Model input.
        y: Target output.
        class_: Model class.

        logdir: Version directory to look underneath from.
        log_name: Name of log directory.
        deep: Search for version directory in depth for non-standard directory hierarchy.
        metric_name: Name of the best metric. Requires checkpoint name format as -
            {anything}{metric_name}={metric_value}.ckpt
        sort_order: Sort order to choose best metric from. Apply only when metric_name is specified.
            Available options - 'min' or 'max'.
    """
    return [evaluate_ckpt(class_=class_, dataset=dataset, ckpt=ckpt, is_graph=is_graph) for ckpt in
            get_checkpoints(logdir=logdir, log_name=log_name, deep=deep, metric_name=metric_name, sort_order=sort_order)]


def evaluate_ckpt(class_: Any, dataset: Any, ckpt: AnyPath, is_graph: bool = True, verbose: bool = False) -> Dict[str, float]:
    """Evaluate checkpoint on x, y.

    Args:
        is_graph: True if dataset is graph
        dataset: Dataset.
        class_: Model class.
        ckpt: Path to checkpoint file.
        verbose: Verbose mode.
    """

    model = class_.load_from_checkpoint(ckpt)
    log_dict = model.hparams.copy()

    pred = []
    y = []

    if is_graph:
        for input in dataset:
            input = torchg.data.batch.Batch.from_data_list([input])
            try:
                pred_ = model.forward(input.x, input.edge_index, input.batch)
            except (TypeError, AttributeError):
                pred_ = model.forward(input.x)

            pred.append(pred_.detach().item())
            y.append(input.y.item())
    else:
        for x, y_ in dataset:
            x.unsqueeze_(dim=0)
            pred_ = model.forward(x)

            pred.append(pred_.item())
            y.append(y_.item())

    x = np.array(pred)
    y = np.array(y)

    log_dict['pcc'], log_dict['pcc_p'] = pearsonr(x, y)
    log_dict['src'], log_dict['src_p'] = spearmanr(x, y)
    log_dict['r2'] = r2_score(y, x)
    log_dict['mse'] = ((y-x) ** 2).sum() / len(x)

    if verbose:
        print()
        print(f'R2: {log_dict["r2"]:.2f}')
        print(f'MSE: {log_dict["mse"]:.2f}')
        print(f'PCC: {log_dict["pcc"]:.2f} ({log_dict["pcc_p"]:.2f})')
        print(f'SRC: {log_dict["src"]:.2f} ({log_dict["src_p"]:.2f})')
        print()

    return log_dict


def graph_saliency_map(model: Any, input: Union[torchg.data.data.Data, torchg.data.batch.Batch], method: str = 'norm')\
        -> torch.Tensor:
    """Calculate saliency map on node features.

    There is no restriction on the model object but only its forward method, i.e.
    model.forward(input.x, input.edge_index, input.batch)

    This function does not support edge features yet.

    Warning: The saliency map only propagates to the input level. If embedding is used, the graph saliency does not
        account for the embedding generator.
    """
    input = input.clone()
    if type(input) == torchg.data.data.Data:
        batch = torch.zeros(input.x.shape[0], dtype=torch.long)
    else:
        batch = input.batch
        if len(batch.unique()) > 1:
            raise ValueError('Only support single-example saliency map.')

    input.x.requires_grad_()
    output = model.forward(input.x, input.edge_index, batch)
    input.x.retain_grad()
    output.backward()
    grad_ = input.x.grad

    if method == 'norm':
        grad = grad_.norm(p=2, dim=1)
    elif method == 'mean':
        grad = grad_.abs().mean(axis=1)
    elif method == 'max':
        grad = grad_.abs().max(axis=1)
    else:
        raise ValueError('Only norm, mean, max supported.')

    return grad


def confusion_matrix_plot(cm: np.array, return_fig: bool = True) -> Union[Figure, Axes]:
    """Utility function get plot confusion matrix ax or figure.

    Args:
        cm: Confusion matrix in numpy.Array.
        return_fig: If true, return matplotlib.Figure. Otherwise, matplotlib.Axes
    """
    display = ConfusionMatrixDisplay(cm)
    _ = display.plot()
    plt.close()
    display.ax_.grid(False)

    if return_fig:
        return display.figure_
    else:
        return display.ax_


def dual_confusion_matrix_plot(cms: List[np.array]) -> Figure:
    """Utility function to plot two confusion matrix in parallel.

    Suggest to use in train-val logging visualization.

    Args:
        cms: List of confusion matrix in numpy.Array
    """
    fig, ax = plt.subplots(figsize=(12 * len(cms) * 12))

    for i, cm in enumerate(cms):
        ax = confusion_matrix_plot(cm, return_fig=False)
        if i == 0:
            fig.axes = [ax]
        else:
            fig.axes.append(ax)

    return fig


def reg_corr_plot(pred: Iterable, y: Iterable, full_range: bool = False) -> plt.Figure:
    """Utility function get seaborn jointplot between prediction and target values.

    Args:
        pred: Predicted values from model.
        y: Target values.
        full_range: If false, restrict plotted data points to true value range.
    """
    df = pd.DataFrame(zip(pred, y), columns=['predicted value', 'true value'])
    if full_range:
        # noinspection PyArgumentList
        lim = (df.values.min(), df.values.max())
    else:
        lim = (df['true value'].min(), df['true value'].max())
    jg = sns.jointplot(data=df, x='predicted value', y='true value', xlim=lim, ylim=lim)
    return jg.fig


def dual_reg_corr_plot(dual_data: Dict[str, Dict[str, torch.Tensor]], full_range: bool = False,
                       margin_ratio: float = 1.1, **kwargs) -> plt.Figure:
    """Utility function to overlay seaborn jointplots of train and validation set.

    Args:
        dual_data: Dictionary of train and val set data in the format of
            {'train': {'pred': predictions, 'true': target values}, 'val': ...}
        full_range: If false, restrict plotted data points to true value range.
        margin_ratio: Extra margin from min-max range.
    """
    # noinspection PyArgumentList
    y_cat = torch.cat((dual_data['train']['y'], dual_data['val']['y']), axis=0)
    # noinspection PyArgumentList
    pred_cat = torch.cat((dual_data['train']['pred'], dual_data['val']['pred']), axis=0)
    df = pd.DataFrame(zip(pred_cat.numpy(), y_cat.numpy()), columns=['predicted value', 'true value'])

    # define plot value range
    if full_range:
        # noinspection PyArgumentList
        lim = (df.values.min(), df.values.max())
    else:
        lim = (df['true value'].values.min(), df['true value'].values.max())

    lim_mid = (lim[0] + lim[1]) / 2
    lim_len = margin_ratio * (lim[1] - lim[0])
    lim = (lim_mid - lim_len / 2, lim_mid + lim_len / 2)
    if 'xlim' not in kwargs:
        kwargs['xlim'] = lim
    if 'ylim' not in kwargs:
        kwargs['ylim'] = lim

    colors = {
        'train': matplotlib.colors.to_rgba('orange', alpha=0.5),
        'val': matplotlib.colors.to_rgba('royalblue', alpha=1)
    }
    df = df.clip(lower=lim[0], upper=lim[1])
    df['hue'] = ['train'] * len(dual_data['train']['y']) + ['val'] * len(dual_data['val']['y'])

    sns.set_theme()
    jg = sns.JointGrid(data=df, x='predicted value', y='true value', hue='hue', **kwargs)
    jg.plot_joint(sns.scatterplot, palette=colors)
    # jg.ax_joint.invert_yaxis()
    legend = jg.ax_joint.get_legend()
    legend.set_title('')
    legend._set_loc(4)  # not a good practice

    common_kwargs = {'stat': 'density', 'alpha': 0.5, 'binwidth': 0.25, 'binrange': kwargs['ylim']}
    sns.histplot(y=dual_data['train']['y'], ax=jg.ax_marg_y, color=colors['train'], **common_kwargs)
    sns.histplot(y=dual_data['val']['y'], ax=jg.ax_marg_y, color=colors['val'], **common_kwargs)

    common_kwargs['binrange'] = kwargs['xlim']
    sns.histplot(x=dual_data['train']['pred'], ax=jg.ax_marg_x, color=colors['train'], **common_kwargs)
    sns.histplot(x=dual_data['val']['pred'], ax=jg.ax_marg_x, color=colors['val'], **common_kwargs)

    jg.set_axis_labels(xlabel='predicted value', ylabel='true value')
    jg.fig.tight_layout()
    return jg.fig


def pdb2distance_map(pdb: AnyPath) -> torch.Tensor:
    """Get the distance map between CA atoms from pdb."""
    pdb = Path(pdb)
    parser = PDB.PDBParser(PERMISSIVE=True, QUIET=True)
    structure = parser.get_structure(pdb.stem, pdb)
    chain = structure[0]['A']

    coors = []
    for resid in range(1, len(chain)+1):
        residue = chain[resid]

        ca = [atom for atom in residue.get_atoms() if atom.get_name() == 'CA']
        assert len(ca) == 1
        ca = ca[0]

        coor = ca.get_vector().get_array()
        coors.append(coor)

    n_residues = len(coors)
    coors = torch.tensor(coors)

    dist_mat = torch.broadcast_to(coors, (n_residues, n_residues, 3))
    dist_mat = dist_mat - dist_mat.transpose(0, 1)
    dist_mat = ((dist_mat ** 2).sum(dim=-1)) ** 0.5
    return dist_mat


def get_corr(x: OrderedIterableType, y: OrderedIterableType) -> CorrType:
    """Given x, y, evaluate PCC, SRC and their p-values."""
    if type(x) is collections.OrderedDict:
        x = list(x.values())
    if type(y) is collections.OrderedDict:
        y = list(y.values())

    src = scipy.stats.spearmanr(x, y)
    corr_dict = {
        'pcc': scipy.stats.pearsonr(x, y),
        'src': (src.correlation, src.pvalue),
    }
    return corr_dict


def plot_corr(x: OrderedIterableType, y: OrderedIterableType, x_name: str, y_name: str, png: Optional[AnyPath] = None,
              rootdir: Optional[AnyPath] = None, **kwargs) -> None:
    """Plot x, y in seaborn jointplot

    Args:
        x: x-axis samples
        y: y-axis samples
        x_name: Name of x variable for plotting
        y_name: Name of y variable for plotting
        png: png filename. Default: {x_name}_{y_name}.png
        rootdir: Destination directory for png.
        kwargs: Keyword arguments for seaborn.jointplot.
    """
    if type(x) is collections.OrderedDict:
        x = list(x.values())
    if type(y) is collections.OrderedDict:
        y = list(y.values())

    assert len(x) == len(y)
    df = pd.DataFrame(zip(x, y), columns=[x_name, y_name])
    jg = sns.jointplot(x=df[x_name], y=df[y_name], **kwargs)

    png = f'{x_name}-{y_name}.png' if png is None else Path(png)
    if rootdir is not None:
        png = Path(rootdir) / png
    plt.savefig(png)
    plt.close()
