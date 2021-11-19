from pathlib import Path, PosixPath, WindowsPath
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import torch
from typing import *
import warnings

from .data import read_log2en, pssm2D
from ..util import plot_corr, get_corr

plt.style.use('ggplot')
AnyPath = Union[str, PosixPath, WindowsPath]
CorrType = Dict[str, Tuple[float, float]]
OrderedIterableType = Union[List, OrderedDict, Tuple]


def visual_pssm_corr(rootdir: Optional[AnyPath] = None, exp_data: Optional[Union[Dict, OrderedDict]] = None,
                     pssm: Optional[AnyPath] = None, **kwargs) -> CorrType:
    """Plot pssm to log2enrichment experiment data and evaluate correlations.

    Args:
        exp_data: Dictionary of mutant name and experiment data, i.e. {mut_name: y}
        pssm: Path to pssm file from psiblast.
        rootdir: Destination directory for exp-pssm.png.
        kwargs: Keyword arguments for seaborn.jointplot.
    """
    if exp_data is None:
        exp_data = read_log2en(return_type='OrderedDict')

    dpssm = pssm2D(pssm=pssm, return_type='OrderedDict', relative=True)  # difference compared to wildtype
    for mut in exp_data.copy():
        if mut not in dpssm:
            warnings.warn(f'{mut} not found in pssm. Skipped.')
            del exp_data[mut]

    dpssm = [dpssm[u] for u in exp_data.keys()]
    exp_data = list(exp_data.values())

    df = pd.DataFrame(zip(exp_data, dpssm), columns=['log2en', 'pssm'])
    jg = sns.jointplot(x=df['pssm'], y=df['log2en'], **kwargs)

    png = 'exp-pssm.png'
    if rootdir is not None:
        png = Path(rootdir) / png
    plt.savefig(png)
    plt.close()

    src = scipy.stats.spearmanr(exp_data, dpssm)
    corr_dict = {
        'pcc': scipy.stats.pearsonr(exp_data, dpssm),
        'src': (src.correlation, src.pvalue),
    }
    return corr_dict


def visual_esm_corr(methods: Optional[Union[str, Tuple]] = None, rootdir: Optional[AnyPath] = None, pdb_name: str = '1gnx',
                    exp_data: Optional[Union[Dict, OrderedDict]] = None, esm_dir: Optional[AnyPath] = None,
                    **kwargs) -> Union[CorrType, Dict[str, CorrType]]:
    """(Legacy) plot esm to log2enrichment experiment data and evaluate correlations.

    Args:
        exp_data: Dictionary of mutant name and experiment data, i.e. {mut_name: y}
        pssm: Path to pssm file from psiblast.
        rootdir: Destination directory for exp-pssm.png.
        kwargs: Keyword arguments for seaborn.jointplot.
    """
    esm_dir = Path('data/embeddings/esm') if esm_dir is None else Path(esm_dir)
    if not esm_dir.exists():
        raise FileNotFoundError(f'{esm_dir} not found.')

    if methods is None:
        methods = ('ave', 'max', 'min')
    elif type(methods) == str:
        methods = [methods]
    else:
        methods = list(methods)

    if exp_data is None:
        exp_data = read_log2en(return_type='OrderedDict')
    esm_values = {method: OrderedDict() for method in methods}

    for mut_name, log2en_value in exp_data.items():
        pt = esm_dir / f'{pdb_name}_{mut_name}.pt'
        esm = torch.load(pt)

        for method in methods:
            if method == 'ave':
                esm_value = esm.mean().item()
            elif method == 'max':
                esm_value = esm.max().item()
            elif method == 'min':
                esm_value = esm.min().item()
            else:
                raise ValueError(f'{method} method not supported.')
            esm_values[method][mut_name] = esm_value

    method_corr = {}
    for method in methods:
        plot_corr(exp_data, esm_values[method], x_name='exp', y_name=f'esm-{method}', rootdir=rootdir, **kwargs)
        method_corr[method] = get_corr(exp_data, esm_values[method])

    if len(methods) == 1:
        return method_corr[methods[0]]
    else:
        return method_corr

