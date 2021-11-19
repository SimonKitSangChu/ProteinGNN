from abc import abstractmethod
from collections import OrderedDict
import multiprocessing as mlp
from numbers import Number
import numpy as np
import os
from pathlib import Path, PosixPath, WindowsPath
import pandas as pd
import pyrosetta
import pytorch_lightning as pl
import re
import shutil
import sys
import torch
import torch.nn.functional as F
import torch_geometric as torchg
from tqdm import tqdm
from typing import List, Union, Optional, Generator, Dict, Tuple, Callable, Set, Type, Iterable, Any
import json
import warnings

AnyNum = Union[int, float]
AnyPath = Union[str, PosixPath, WindowsPath]
AtomType = pyrosetta.rosetta.core.conformation.Atom
ResidueType = pyrosetta.rosetta.core.conformation.Residue
ConformationType = pyrosetta.rosetta.core.conformation.Conformation
PoseType = pyrosetta.rosetta.core.pose.Pose
AtomIDType = pyrosetta.rosetta.core.id.AtomID
EdgeType = List[int]

VERBOSE = 0
CLONE_RMSD = 1e-7
AA_ALPHABETS = list('ACDEFGHIKLMNPQRSTVWY')
PYROSETTA_INIT = False
ESM_MODEL = None
ESM_ALPHABET = None
ESM_BATCH_CONVERTER = None
CUDA_AVAILABLE = torch.cuda.is_available()


def is_dummy_mut(mut_name: str) -> bool:
    return mut_name[0] == mut_name[-1]


def aa2index(aa: str) -> int:
    aa = aa.capitalize()
    assert aa in AA_ALPHABETS, f'{aa} is not supported.'
    return AA_ALPHABETS.index(aa)


def seq2index(seq: Iterable) -> List[int]:
    return [aa2index(aa) for aa in seq]


def seq2onehot(seq: Iterable) -> torch.Tensor:
    index = torch.tensor(seq2index(seq))
    return F.one_hot(index, num_classes=len(AA_ALPHABETS))


def read_pssm(pssm: AnyPath, return_type: str = 'DataFrame', relative: bool = False):
    if return_type not in ('DataFrame', 'Array', 'Tensor', 'OrderedDict'):
        raise ValueError('Only DataFrame, Array, Tensor and OrderedDict supported.')

    pssm = Path(pssm)
    df = pd.read_csv(pssm, sep='\s+', skiprows=3, skipfooter=5, engine='python')
    df.loc[len(df)] = df.columns
    df = df.iloc[:, :22]

    df.columns = ['resid', 'wt'] + list('ARNDCQEGHILKMFPSTWYV')
    df = df[['resid', 'wt'] + AA_ALPHABETS]
    df = df.astype({aa: float for aa in AA_ALPHABETS})

    df = df.astype({'resid': int}).sort_values('resid')
    df = df.set_index('resid')

    if relative:
        for i, row in df.copy().iterrows():
            wt_v = row[row['wt']]
            df.loc[i, AA_ALPHABETS] = row[AA_ALPHABETS] - wt_v

    if return_type == 'DataFrame':
        return df
    elif return_type == 'Array':
        return df[AA_ALPHABETS].values
    elif return_type == 'Tensor':
        return torch.from_numpy(df[AA_ALPHABETS].values)
    else:
        pssm = OrderedDict()
        for resid in df.index:
            wt_aa = df.loc[resid, 'wt']
            for aa in AA_ALPHABETS:
                pssm[f'{wt_aa}{resid}{aa}'] = df.loc[resid, aa]

        return pssm


def pssm1D(seq: Iterable, pssm=None, return_type='Array', **kwargs):
    """Obtain pssm given sequence."""
    if pssm is None:
        pssm = read_pssm(return_type=return_type, **kwargs)

    pssm_values = [pssm[i, aa2index(aa)] for i, aa in enumerate(seq)]
    return torch.tensor(pssm_values).unsqueeze(-1)


def pssm2D(*args, return_type='Tensor', **kwargs):
    """Alias of read_pssm."""
    return read_pssm(*args, return_type=return_type, **kwargs)


def get_esm_representations(data, layers: Optional[List[int]] = None, model_name='esm1b_t33_650M_UR50S'):
    # import esm and create object only when necessary to save memory
    # but globals is a bad habit
    model = globals()['ESM_MODEL']
    alphabet = globals()['ESM_ALPHABET']
    batch_converter = globals()['ESM_BATCH_CONVERTER']

    if model is None or alphabet is None or batch_converter is None:
        model, alphabet = getattr(esm.pretrained, model_name)()
        batch_converter = alphabet.get_batch_converter()

        model = model.eval()
        if CUDA_AVAILABLE:
            model = model.cuda()

        globals()['ESM_MODEL'] = model
        globals()['ESM_ALPHABET'] = alphabet
        globals()['ESM_BATCH_CONVERTER'] = batch_converter

    if layers is None:
        layers = [33]

    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    with torch.no_grad():
        if CUDA_AVAILABLE:
            batch_tokens = batch_tokens.cuda()
        results = model(batch_tokens, repr_layers=layers)

    representations = [results['representations'][layer_i].squeeze()[1:-1] for layer_i in layers]
    representations = torch.cat(representations, dim=1).squeeze()
    return representations.cpu()


def get_directory_by_suffix(src_dir: AnyPath, suffixes: List, deep: bool = False) -> Generator[Path, None, None]:
    """Return files matching suffix (with .) underneath a directory (optionally recursive)."""
    src_dir = Path(src_dir)

    if deep:
        for rootdir, subdirs, files in os.walk(src_dir):
            for file in files:
                if Path(file).suffix in suffixes:
                    yield Path(rootdir) / file
    else:
        if len(suffixes) == 1:
            return src_dir.glob(f'*{suffixes[0]}')
        else:
            for file in src_dir.glob('*'):
                if file.suffix in suffixes:
                    yield file


def get_directory_pdb(src_dir: Path, deep: bool = False) -> Generator[Path, None, None]:
    """Return all pdb (Path) underneath a directory (optionally recursive)."""
    return get_directory_by_suffix(src_dir, ['.pdb'], deep)


def get_pyrosetta_flags(flags: Optional[AnyPath] = None, params_dir: Optional[AnyPath] = None,
                        verbose_level: int = 300 if VERBOSE else 0):
    """Return pyrosetta initialize flags with ligand parameters and verbose level in string.

    If the ligand name clashes with existing ones in database, consider to rename/comment out
    in database/chemical/residue_type_sets/fa_standard/residue_types.txt
    """
    if not flags:
        flags = 'predataset/flags'
    if not params_dir:
        params_dir = 'predataset/Ligand_params'

    flags = Path(flags)
    with flags.open('r') as fopen:
        flags_str = fopen.read()

    params_dir = Path(params_dir)
    params = params_dir.glob('*.params')
    params = sorted(map(str, params))
    if not params:
        raise FileNotFoundError(f'No params under {params_dir}')
    flags_str += '\n-extra_res_fa ' + ' '.join(params)

    flags_str += f'\n-out::level {verbose_level}'
    return flags_str


def create_dir(path: AnyPath, overwrite: bool = False) -> bool:
    """Create (and overwrite) path directory."""
    path = Path(path)
    if overwrite:
        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)
    return True


def pdb2pose_atoms(pdb: AnyPath) -> Tuple[PoseType, Dict[AtomIDType, AtomType]]:
    if not Path(pdb).exists():
        raise FileNotFoundError(f'{pdb} not found in graph processing.')
    pose = pyrosetta.pose_from_pdb(str(pdb))
    atoms = {}

    for residue in pose.residues:
        resid = residue.seqpos()

        for atom_idx in range(1, residue.natoms() + 1):
            atom = residue.atom(atom_idx)
            atom_id = pyrosetta.rosetta.core.id.AtomID(atom_idx, resid)
            atoms[atom_id] = atom

    return pose, atoms


def atom_id2atom_name(atom_id: AtomIDType, pose: PoseType) -> str:
    residue = pose.residue(atom_id.rsd())
    atom_name = residue.atom_name(atom_id.atomno())
    atom_name = atom_name.strip()
    return atom_name


def sort_atom_ids(atom_ids: Iterable[AtomIDType]) -> List[AtomIDType]:
    """Sort AtomIDs by (residue id, atom number)"""

    def atom_id2int(atom_id):
        return atom_id.rsd(), atom_id.atomno()

    return sorted(atom_ids, key=atom_id2int)


def _parameter2config(instance: Any, parameters: Dict[str, Any]) -> Dict[str, Any]:
    class_name = instance.__class__.__name__
    return {'class_name': class_name, 'parameters': parameters}


class BaseNodeFilter:
    """Base class for node (atom) filter.

    Only filters virtual atoms.
    """

    def __init__(self, name: Optional[str] = None):
        if name is None:
            name = self.__class__.__name__
        self.name = name

    @property
    def config(self) -> Dict[str, Any]:
        parameters = {'name': self.name}
        return _parameter2config(self, parameters)

    def filter_func(self, pose: PoseType, atom_id: AtomIDType, atom_name: str) -> bool:
        return True

    def filter_virt(self, pose: PoseType, atom_id: AtomIDType, atom_name: str) -> bool:
        """Filter out virtual atoms."""
        name_split = re.findall(r'[^\W\d_]+|\d+', atom_name)
        if len(name_split) >= 2:
            if 'V' == name_split[-2]:
                return False

        return True

    def filter(self, pose: PoseType, atoms: Dict[AtomIDType, AtomType]) -> Dict[AtomIDType, AtomType]:
        """Filter by arbitrary method. Filter virtual atoms regardless."""
        new_atoms = {}

        for atom_id, atom in atoms.items():
            atom_name = atom_id2atom_name(atom_id, pose)
            if self.filter_func(pose, atom_id, atom_name) and self.filter_virt(pose, atom_id, atom_name):
                new_atoms[atom_id] = atom

        return new_atoms


# TODO filter by chain names
class ChainNodeFilter(BaseNodeFilter):
    """Filter node by chain"""

    def __init__(self, chain_names: Union[Tuple, List], name: Optional[str] = None):
        if not chain_names:
            raise ValueError('chain_names is empty.')

        self.chain_names = list(chain_names).copy()
        super().__init__(name=name)

    @property
    def config(self) -> Dict[str, Dict]:
        parameters = {'name': self.name, 'chain_names': self.chain_names.copy()}
        return _parameter2config(self, parameters)

    def filter_func(self, pose: PoseType, atom_id: AtomIDType, atom_name: str) -> bool:
        """Filter node by chain_names"""
        raise NotImplementedError  # how to get chain name instead of chain id in pyrosetta?


# TODO filter by backbone atoms
# class BackboneNodeFilter(BaseNodeFilter)

# TODO filter by residue name (3 letters)
# class ResidueNameFilter(BaseNodeFilter)

class AtomNameNodeFilter(BaseNodeFilter):
    """Filter node by atom name"""

    def __init__(self, atom_name_pass: List = None, name: Optional[str] = None):
        if type(atom_name_pass) != list:
            raise TypeError('Either atom_name_pass is not List.')

        self.atom_name_pass = atom_name_pass.copy()
        super().__init__(name=name)

    @property
    def config(self) -> Dict[str, Dict]:
        parameters = {'name': self.name, 'atom_name_pass': self.atom_name_pass.copy()}
        return _parameter2config(self, parameters)

    def filter_func(self, pose: PoseType, atom_id: AtomIDType, atom_name: str) -> bool:
        """Filter node by atom_name_pass"""
        return atom_name.strip() in self.atom_name_pass


class CompositeNodeFilter(BaseNodeFilter):
    """Composite of NodeFilter(s)."""

    def __init__(self, components: Optional[Iterable[Type[BaseNodeFilter]]] = None,
                 intersection: bool = True, name: Optional[str] = None):
        """Initialize with name and any number of NodeFilter(s).

        Args:
            components: Iterable of NodeFilters
            intersection: Filter if it does not fulfill all node filter criteria
        """
        if not components:
            raise ValueError('No filter received.')
        if not all(issubclass(filter.__class__, BaseNodeFilter) for filter in components):
            raise TypeError('Some filter(s) is not subclass of BaseNodeFilter.')

        super().__init__(name=name)
        self.components = components
        self.intersection = intersection

    @property
    def config(self) -> List[Dict]:
        config = [component.config for component in self.components]
        parameters = {'name': self.name, 'intersection': self.intersection}
        subconfig = _parameter2config(self, parameters)
        config.append(subconfig)

        return config

    def filter_func(self, pose: PoseType, atom_id: AtomIDType, atom_name: str) -> bool:
        """Filter by union/intersection of components"""
        bools = (component.filter_func(pose, atom_id, atom_name) for component in self.components)
        return all(bools) if self.intersection else any(bools)


class BaseNodeFeaturizer:
    """Base class for node featurization."""

    def __init__(self, name: Optional[str] = None):
        self.name = name

    @property
    def config(self) -> Dict[str, Dict]:
        raise NotImplementedError

    @abstractmethod
    def featurize(self, pose: PoseType, atom_id: AtomIDType) -> List[Number]:
        raise NotImplementedError


# TODO featurize by residue name (3 letters)
# class ResidueNodeFeaturizer(BaseNodeFeaturizer)


class AtomtypeNodeFeaturizer(BaseNodeFeaturizer):
    """Featurize node by atom type"""

    def __init__(self, atomtypes: Optional[Dict[str, list]] = None, atom_cats: Optional[Dict[str, int]] = None,
                 name: Optional[str] = None):
        """Initialize with atom types and atom type categories.

        As a safety precaution, all atom types must be contained in atom categories. Provide
        custom atomtypes and atom_cats for 'AnythingElse' atom type if necessary. See example
        below.

        Args:
            atomtypes: atom type containing a set of atom names. For example,
                {'C': {'C', 'CA', 'CB', 'CD', 'CD1'}, 'O': {'O', 'OD1', 'OD2'}}
            atom_cats: atom type mapping to numerical index. INDEX SHOULD START AT 1.
            For example,
                {'C': 1, 'H': 2, 'O': 3, 'N': 4, 'S': 5, 'MG': 6}
        """
        self._atom_cats = None
        self._atomtypes = None
        self._atomname2cat = None

        if atom_cats is None:
            atom_cats = {'C': 1, 'H': 2, 'O': 3, 'N': 4, 'S': 5, 'MG': 6}
        assert 0 not in atom_cats.keys(), '0 is not allowed in atom_cats.'

        self.atom_cats = atom_cats

        if atomtypes is None:
            atomtypes = {
                'N': ['N', 'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NV', 'NZ'],
                'S': ['SD', 'SG'],
                'O': ['O', 'OD1', 'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH', 'OXT'],
                'C': ['C', 'CA', 'CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3', 'CG', 'CG1', 'CG2', 'CH2', 'CZ',
                      'CZ2', 'CZ3'],
                'H': ['1H', '1HA', '1HB', '1HD', '1HD1', '1HD2', '1HE', '1HE2', '1HG', '1HG1', '1HG2', '1HH1', '1HH2',
                      '1HZ', '2H', '2HA', '2HB', '2HD', '2HD1', '2HD2', '2HE', '2HE2', '2HG', '2HG1', '2HG2', '2HH1',
                      '2HH2', '2HZ', '3H', '3HB', '3HD1', '3HD2', '3HE', '3HG1', '3HG2', '3HZ', 'H', 'HA', 'HB', 'HD1',
                      'HD2', 'HE', 'HE1', 'HE2', 'HE3', 'HG', 'HG1', 'HH', 'HH2', 'HZ', 'HZ2', 'HZ3'],
                'MG': ['MG']
            }  # JSON serializable
        self.atomtypes = atomtypes

        super().__init__(name=name)

    @property
    def config(self) -> Dict[str, Dict]:
        parameters = {
            'name': self.name,
            'atom_cats': self.atom_cats,
            'atomtypes': self.atomtypes
        }
        return _parameter2config(self, parameters)

    def _update_atomname2cat(self) -> bool:
        """Update atomname2cat upon change in atom_cats and atomtypes"""
        mis_atomtypes = set()

        if not self._atomtypes:
            return False

        self._atomname2cat = {}
        for key, values in self._atomtypes.items():
            if key not in self._atom_cats:
                mis_atomtypes.add(key)

            for value in values:
                self._atomname2cat[value] = self._atom_cats[key]

        if mis_atomtypes:
            raise KeyError(f'{mis_atomtypes} not found in atom_cats.')

        return True

    @property
    def atom_cats(self) -> Dict[str, int]:
        return self._atom_cats.copy()

    @atom_cats.setter
    def atom_cats(self, atom_cats: Dict[str, int]):
        values = list(atom_cats.values())
        if len(values) != len(set(values)):
            raise ValueError('Duplicate atom_cats indices are not allowed.')

        self._atom_cats = atom_cats.copy()
        self._update_atomname2cat()

    @property
    def atomtypes(self) -> Dict[str, Dict]:
        return self._atomtypes.copy()

    @atomtypes.setter
    def atomtypes(self, atomtypes: Dict[str, Dict]):
        self._atomtypes = atomtypes.copy()
        self._update_atomname2cat()

    @property
    def n_atom_types(self) -> int:
        return len(self.atom_cats)

    @property
    def atomname2cat(self) -> Dict[str, int]:
        return self._atomname2cat.copy()

    def featurize(self, pose: PoseType, atom_id: AtomIDType) -> List[Number]:
        """Annotate node by atom category in one-hot encoding"""
        residue = pose.residue(atom_id.rsd())
        atom_name = residue.atom_name(atom_id.atomno())
        atom_name = atom_name.strip()
        atom_cat = self._atomname2cat[atom_name]

        # index must start at 1
        encoding = [1 if atom_cat == i else 0
                    for i in range(1, len(self.atom_cats) + 1)]
        return encoding


class SeqEmbNodeFeaturizer(BaseNodeFeaturizer):
    """Featurize node by (external) sequence embedding."""

    def __init__(self, emb_dir: AnyPath, name: Optional[str] = None):
        """Initialize with directory holding sequence embeddings.

        WARNING: MAKE SURE PDB NUMBERING MATCHES THAT ON RCSB FASTA!

        All embeddings must be named in format '{pdb_code}.pt'. Each of them should contain
        a tensor of the sequence embedding in shape (seq_dim, emb_dim).

        Args:
            emb_dir: Embedding directory
        """
        emb_dir = Path(emb_dir)
        if not emb_dir.exists():
            raise FileNotFoundError(f'{emb_dir} not found.')

        super().__init__(name=name)
        self.emb_dir = emb_dir
        self.pt_dict = {pt.stem: pt for pt in emb_dir.glob('*.pt')}
        if not self.pt_dict:
            raise FileNotFoundError('No embedding pt file found underneath directory.')

        warnings.warn('Make sure pdb numbering matches rcsb fasta!')  # leap of faith!

    def get_emb(self, pdb_name: str):
        """Get embedding by pdb name"""
        pt = self.pt_dict[pdb_name]
        emb = torch.load(str(pt))
        assert len(emb.shape) == 2, 'Sequence embedding shape should be in (seq_dim, emb_dim).'
        return emb

    @property
    def config(self) -> Dict[str, Dict]:
        parameters = {
            'name': self.name,
            'emb_dir': str(self.emb_dir),
        }
        return _parameter2config(self, parameters)

    def featurize(self, pose: PoseType, atom_id: AtomIDType) -> List[Number]:
        """Annotate node by sequence embedding."""
        pdb = pose.pdb_info().name()
        pdb_name = Path(pdb).stem
        emb = self.get_emb(pdb_name)

        rsd = atom_id.rsd()
        resid = pose.pdb_info().pose2pdb(rsd)

        resid = int(resid.split()[0])
        return list(emb[resid-1])


class CompositeNodeFeaturizer(BaseNodeFeaturizer):
    """Composite of NodeFeaturizer(s)."""

    def __init__(self, components: Optional[List[Type[BaseNodeFeaturizer]]] = None, name: Optional[str] = None):
        """Initialize with name and any number of NodeFeaturizer(s).

        Args:
            components: (Ordered) list of NodeFeaturizers
            intersection: Edge determination by intersection of featurizers
        """
        if not components:
            raise ValueError('No featurizer received.')
        if not all(issubclass(featurizer.__class__, BaseNodeFeaturizer) for featurizer in components):
            raise TypeError('Some featurizer(s) is not subclass of NodeFeaturizer.')

        super().__init__(name)
        self.components = list(components)  # mutable and ordered
        self.n_features = None

    @property
    def config(self) -> List[Dict]:
        config = [component.config for component in self.components]
        parameters = {'name': self.name}
        subconfig = _parameter2config(self, parameters)
        config.append(subconfig)

        return config

    def featurize(self, pose: PoseType, atom_id: AtomIDType) -> List[Number]:
        """Annotate in accordance to sub-NodeFeaturizers and append outputs."""
        features = []
        for component in self.components:
            feature = component.featurize(pose, atom_id)
            features += feature

        if self.n_features:
            assert len(features) == self.n_features, \
                f'Irregular feature length. (storage: {self.n_features} v.s. current {len(features)})'
        else:
            self.n_features = len(features)

        return features


class BaseEdgeFeaturizer:
    """Base class of edge featurization (including edge determination)."""

    def __init__(self, is_edge_only: bool = False, name: Optional[str] = None):
        """
        Initialize and optionally allow edge determination only (empty feature list).
        """
        if name is None:
            name = self.__class__.__name__
        self.name = name
        self.is_edge_only = is_edge_only

    @property
    def config(self) -> Dict[str, Dict]:
        raise NotImplementedError

    @abstractmethod
    def featurize(self, pose: PoseType, atom_id1: AtomIDType, atom_id2: AtomIDType) -> Tuple[bool, List[Number]]:
        raise NotImplementedError


class BondedEdgeFeaturizer(BaseEdgeFeaturizer):
    """Featurize edge by chemical bond formation."""

    def __init__(self, is_edge_only: bool = False, name: Optional[str] = None):
        super().__init__(is_edge_only=is_edge_only, name=name)

    @property
    def config(self) -> Dict[str, Dict]:
        parameters = {'name': self.name, 'is_edge_only': self.is_edge_only}
        return _parameter2config(self, parameters)

    def featurize(self, pose: PoseType, atom_id1: AtomIDType, atom_id2: AtomIDType) -> Tuple[bool, List[Number]]:
        """Annotate edge with [1] if chemically bonded. Otherwise [0]."""
        is_edge = pose.conformation().is_bonded(atom_id1, atom_id2)
        if self.is_edge_only:
            return is_edge, []

        feature = [1] if is_edge else [0]
        return is_edge, feature


class DistanceEdgeFeaturizer(BaseEdgeFeaturizer):
    """Featurize edge by atom separation distance."""

    def __init__(self, is_edge_only: bool = False, max_distance: Number = 0., sigma: float = 0., name: Optional[str] = None):
        """Initialize with max separation distance."""
        super().__init__(is_edge_only=is_edge_only, name=name)
        self.max_distance = max_distance
        self.sigma = sigma

    @property
    def config(self) -> Dict[str, Dict]:
        parameters = {'name': self.name, 'is_edge_only': self.is_edge_only, 'sigma': self.sigma,
                      'max_distance': self.max_distance}
        return _parameter2config(self, parameters)

    def featurize(self, pose: PoseType, atom_id1: AtomIDType, atom_id2: AtomIDType) -> Tuple[bool, List[Number]]:
        """Annotate edge with [1] if nodes (atoms) are closer than max_distance. Otherwise [0]."""
        xyz1 = pose.xyz(atom_id1)
        xyz2 = pose.xyz(atom_id2)

        xyz1 = np.asarray(xyz1)
        xyz2 = np.asarray(xyz2)
        if self.sigma:
            xyz1 += np.random.normal(0, self.sigma / 3 ** 0.5, 3)  # sigma in r -> (x, y, z)
            xyz2 += np.random.normal(0, self.sigma / 3 ** 0.5, 3)

        distance = np.linalg.norm(xyz1 - xyz2)

        is_edge = distance < self.max_distance
        if self.is_edge_only:
            return is_edge, []

        feature = [1] if is_edge else [0]
        return is_edge, feature


class HbondEdgeFeaturizer(BaseEdgeFeaturizer):
    """Featurize edge by hydrogen bond."""

    def __init__(self, is_edge_only: bool = False, name: Optional[str] = None, **kwargs):
        """Initialize with keyword argument(s) for pyrosetta HBondSet function."""
        super().__init__(is_edge_only=is_edge_only, name=name)
        self._pose = None
        self._hbond_set = None
        self._n_hbonds = None
        # self._atom_id_set = None
        self._atom_pair_str_set = None

        self.hbond_set_kwargs = kwargs
        if 'bb_only' not in self.hbond_set_kwargs:  # avoid pyrosetta default value trap
            self.hbond_set_kwargs['bb_only'] = False

    @property
    def config(self):
        parameters = {'name': self.name, 'hbond_set_kwargs': self.hbond_set_kwargs}
        return _parameter2config(self, parameters)

    @property
    def pose(self) -> PoseType:
        if self._pose is None:
            raise ValueError('No pose in featurizer yet.')
        return self._pose.clone()

    @pose.setter
    def pose(self, pose) -> None:
        self._pose = pose.clone()
        self._hbond_set = pyrosetta.rosetta.core.scoring.hbonds.HBondSet(pose, **self.hbond_set_kwargs)
        self._n_hbonds = self._hbond_set.nhbonds()
        self._atom_pair_str_set = set()

        # note: does not support directional edge (donor -> acceptor)
        for i_hbond in range(1, self._n_hbonds + 1):
            hbond = self._hbond_set.hbond(i_hbond)
            don_atm = hbond.don_hatm()
            don_res = hbond.don_res()
            don_atom_id = pyrosetta.rosetta.core.id.AtomID(don_atm, don_res)

            acc_atm = hbond.acc_atm()
            acc_res = hbond.acc_res()
            acc_atom_id = pyrosetta.rosetta.core.id.AtomID(acc_atm, acc_res)

            atom_pair_str = str(don_atom_id) + str(acc_atom_id)
            self._atom_pair_str_set.add(atom_pair_str)

    @pose.deleter
    def pose(self) -> None:
        self._pose = None
        self._hbond_set = None
        self._n_hbonds = None
        # self._atom_id_set = None
        self._atom_pair_str_set = None

    def featurize(self, pose: PoseType, atom_id1: AtomIDType, atom_id2: AtomIDType) -> Tuple[bool, List[Number]]:
        """Annotate edge with [1] if hydrogen bonded (defined by pyrosetta). Otherwise [0].

        Directionality (donor -> acceptor) not supported.
        """
        if self._pose is None:
            self.pose = pose
        elif pyrosetta.rosetta.core.scoring.all_atom_rmsd_nosuper(self._pose, pose) > CLONE_RMSD:
            self.pose = pose

        is_edge = (str(atom_id1) + str(atom_id2) in self._atom_pair_str_set) \
                  or (str(atom_id2) + str(atom_id1) in self._atom_pair_str_set)
        if self.is_edge_only:
            return is_edge, []

        feature = [1] if is_edge else [0]
        return is_edge, feature


class CompositeEdgeFeaturizer(BaseEdgeFeaturizer):
    """Composite of EdgeFeaturizer(s)."""

    def __init__(self, intersection: bool = False, components: Optional[List[Type[BaseEdgeFeaturizer]]] = None,
                 name: Optional[str] = None):
        """Initialize with name and any number of EdgeFeaturizer(s).

        Args:
            components: (Ordered) list of EdgeFeaturizers
            intersection: Edge determination by intersection of featurizers
        """
        if not components:
            raise ValueError('No featurizer received.')
        if not all(issubclass(featurizer.__class__, BaseEdgeFeaturizer) for featurizer in components):
            raise TypeError('Some featurizer(s) is not subclass of BaseEdgeFeaturizer.')

        super().__init__(name=name)
        self.components = list(components)  # mutable and ordered
        self.intersection = intersection
        self.n_features = None

    @property
    def config(self) -> List[Dict]:
        config = [component.config for component in self.components]
        parameters = {'name': self.name, 'intersection': self.intersection}
        subconfig = _parameter2config(self, parameters)
        config.append(subconfig)

        return config

    def featurize(self, pose: PoseType, atom_id1: AtomIDType, atom_id2: AtomIDType) -> Tuple[bool, List[Number]]:
        """Annotate in accordance to sub-EdgeFeaturizers and append outputs."""
        is_edges = []
        features = []
        for featurizer in self.components:
            is_edge, feature = featurizer.featurize(pose, atom_id1, atom_id2)
            is_edges.append(is_edge)
            features += feature

        is_edge = all(is_edges) if self.intersection else any(is_edges)
        if self.n_features:
            assert len(features) == self.n_features, \
                f'Irregular feature length. (storage: {self.n_features} v.s. current {len(features)})'
        else:
            self.n_features = len(features)

        return is_edge, features


class DatasetFactory:
    """Class for dataset creation."""

    def __init__(self, predataset_path: Optional[AnyPath] = None, dataset_path: Optional[AnyPath] = None,
                 name: Optional[str] = None, mutant_y: Union[Dict, OrderedDict, None] = None):
        """Initialize with (pre-)dataset location."""

        if name is None:
            name = self.__class__.__name__
        self.name = name
        self.mutant_y = mutant_y.copy()

        self._node_filter = None
        self._node_featurizer = None
        self._edge_featurizer = None

        if predataset_path is None:
            self.predataset_path = 'predataset'
        else:
            self.predataset_path = Path(predataset_path)

        if dataset_path is None:
            self.dataset_path = 'dataset'
        else:
            self.dataset_path = Path(dataset_path)

        self._shared_pose = None  # same pdb/pose speed-up
        self._shared_atoms = None

    @classmethod
    def create_from_json(cls, path: Optional[AnyPath] = None):
        """Create instance from json configuration.

        See create_from_dict for details.
        """
        if path is None:
            path = 'dataset/config.json'

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f'Config json {path} not found.')

        with path.open('r') as f:
            config = json.load(f)

        return cls.create_from_config(config)

    @classmethod
    def create_from_config(cls, config_dict: Dict[Any, Any]):
        """Create instance from configuration dictionary.

        Args:
            config_dict: Nested configuration dictionary
            {
                'class_name': class name (not used),
                'parameters': {
                    'name': name of DatasetFactory instance,
                    **kwargs: keywaord arguments for DatasetFactory
                    },

                'node_filter': {[
                    'class_name': filter class name, e.g. AtomNameNodeFilter,
                    'parameters': {
                        'name': name of filter class instance,
                        **kwargs: keyword arguments for filter class
                    },
                    (optional) another class for composite filter...
                    ...
                    (optional) composite filter
                    'class_name': composite filter class name (not used),
                    'parameters': {
                        'name': name of composite filter class instance,
                        **kwargs: keyword arguments for composite filter class
                    }
                ]},
                node_featurizer: {...}
                edge_featurizer: {...}
            }
        """
        for key in ('parameters', 'node_filter', 'node_featurizer', 'edge_featurizer'):
            if key not in config_dict:
                raise KeyError(f'{key} not found in json.')

        instance = cls(**config_dict['parameters'])
        composite_classes = {
            'node_filter': CompositeNodeFilter,
            'node_featurizer': CompositeNodeFeaturizer,
            'edge_featurizer': CompositeEdgeFeaturizer,
        }

        for attribute_name, composite_class in composite_classes.items():
            attribute_parameters = config_dict[attribute_name]
            if not attribute_parameters:
                raise ValueError(f'Empty {attribute_name} parameters.')

            elif len(attribute_parameters) == 1:
                attribute_parameters = attribute_parameters[0]
                attributer = getattr(sys.modules[__name__], attribute_parameters['class_name'])
                attributer = attributer(**attribute_parameters['parameters'])
                setattr(instance, attribute_name, attributer)

            elif len(attribute_parameters) == 2:
                raise ValueError(f'Parameter set count cannot be 2 with composite {attribute_name}.')

            else:
                subattributers = []
                for subattribute_parameters in attribute_parameters[:-1]:
                    subattributer = getattr(sys.modules[__name__], subattribute_parameters['class_name'])
                    subattributer = subattributer(**subattribute_parameters['parameters'])
                    subattributers.append(subattributer)

                subattribute_parameters = attribute_parameters[-1]
                attributer = composite_classes[attribute_name](
                    components=subattributers,
                    **subattribute_parameters['parameters']
                )
                setattr(instance, attribute_name, attributer)

        return instance

    @property
    def config(self) -> Dict[str, Any]:
        parameters = {
            'name': self.name,
            'predataset_path': str(self.predataset_path),
            'dataset_path': str(self.dataset_path)
        }
        config_dict = _parameter2config(self, parameters)

        for attribute_name in ('node_filter', 'node_featurizer', 'edge_featurizer'):
            config = getattr(self, f'_{attribute_name}').config
            assert type(config) in (dict, list), f'node_filter configuration is not type dict nor list.'

            if type(config) == dict:
                config_dict[attribute_name] = [config]
            else:
                config_dict[attribute_name] = config

        return config_dict

    def dump_config(self, path: Optional[AnyPath] = None, overwrite: bool = False):
        if path is None:
            dataset_path = self.dataset_path
            create_dir(dataset_path)
            path = dataset_path / 'config.json'

        path = Path(path)
        if path.exists() and overwrite:
            path.unlink()

        if path.exists() and not overwrite:
            raise FileExistsError(f'Config json {path} exists under no overwrite option.')

        json.dump(self.config, path.open('w'))

    @property
    def node_filter(self) -> Type[BaseNodeFilter]:
        if self._node_filter is None:
            raise ValueError('No node_filter was set.')
        return self._node_filter

    @node_filter.setter
    def node_filter(self, node_filter) -> None:
        if issubclass(node_filter.__class__, BaseNodeFilter):
            self._node_filter = node_filter
        else:
            raise TypeError('node_filter must be subclass of BaseNodeFilter.')

    @node_filter.deleter
    def node_filter(self) -> None:
        self._node_filter = None

    @property
    def node_featurizer(self) -> Type[BaseNodeFeaturizer]:
        if self._node_featurizer is None:
            raise ValueError('No node_featurizer was set.')
        return self._node_featurizer

    @node_featurizer.setter
    def node_featurizer(self, node_featurizer) -> None:
        if issubclass(node_featurizer.__class__, BaseNodeFeaturizer):
            self._node_featurizer = node_featurizer
        else:
            raise TypeError('node_featurizer must be subclass of BaseNodeFeaturizer.')

    @node_featurizer.deleter
    def node_featurizer(self) -> None:
        self._node_featurizer = None

    @property
    def edge_featurizer(self) -> Type[BaseEdgeFeaturizer]:
        if self._edge_featurizer is None:
            raise ValueError('No edge_featurizer was set.')
        return self._edge_featurizer

    @edge_featurizer.setter
    def edge_featurizer(self, edge_featurizer) -> None:
        if issubclass(edge_featurizer.__class__, BaseEdgeFeaturizer):
            self._edge_featurizer = edge_featurizer
        else:
            raise TypeError('edge_featurizer must be subclass of BaseEdgeFeaturizer.')

    @edge_featurizer.deleter
    def edge_featurizer(self) -> None:
        self._edge_featurizer = None

    def get_y(self, pdb: AnyPath):
        mut_name = pdb.stem.split('_')[-1]
        return torch.tensor(self.mutant_y[mut_name], dtype=torch.float)

    def _filter_atoms(self, pose: PoseType, atoms: Dict[AtomIDType, AtomType]) -> Dict[AtomIDType, AtomType]:
        return self._node_filter.filter(pose, atoms)

    def _node_feature(self, pose: PoseType, atom_id: AtomIDType) -> List[Number]:
        return self._node_featurizer.featurize(pose, atom_id)

    def _edge_feature(self, pose: PoseType, atom_id1: AtomIDType, atom_id2: AtomIDType) -> List[Number]:
        return self._edge_featurizer.featurize(pose, atom_id1, atom_id2)

    def set_shared(self, pdb: AnyPath):
        self._shared_pose, self._shared_atoms = pdb2pose_atoms(pdb)

    def process_graph(self, pdb: AnyPath, pos_flag: bool = False, shared_pose: bool = False,
                      data_attr_dict: Optional[Dict[str, Callable]] = None) -> torchg.data.Data:
        """Process pdb structure in pyrosetta and return sample data as undirected graph.

        Args:
            pdb: path to pdb.
            pos_flag: include pos (xyz) in node (default: False).
            shared_pose: assume all pdbs are the same and reuse pose to speed up
            data_attr_dict: functions to annotate data (sample). Key must be attribute name.
                Value is any function with arguments (pose, pdb). For example,
                {'dummy': lambda x: 'dummy'} will get you data.dummy = 'dummy'
        """
        if shared_pose and (self._shared_pose is None or self._shared_atoms is None):
            raise ValueError('shared_pose is True but no self._shared_pose found. '
                             'User forgot to set_shared?')

        if shared_pose:
            pose = self._shared_pose
            atoms = self._shared_atoms
        else:
            pose, atoms = pdb2pose_atoms(pdb)
            atoms = self._filter_atoms(pose, atoms)

        atom_ids = sort_atom_ids(atoms)

        x = []  # size = (num_nodes, num_node_features)
        edge_index = []  # size = (2, num_edges)
        edge_attr = []  # size = (num_edges, num_edge_features)
        pos = []  # size = (num_nodes, num_dimensions)

        for i, atom_id1 in enumerate(atom_ids):
            x.append(self._node_feature(pose, atom_id1))
            if pos_flag:
                coord1 = pose.xyz(atom_id1)
                pos.append(coord1)

            for j, atom_id2 in enumerate(atom_ids[i + 1:], i + 1):
                is_edge, edge_feature = self._edge_feature(pose, atom_id1, atom_id2)
                if is_edge:
                    edge = [i, j]
                    edge_index.append(edge)
                    if edge_feature:
                        edge_attr.append(edge_feature)

                    edge = [j, i]
                    edge_index.append(edge)
                    if edge_feature:
                        edge_attr.append(edge_feature)

        x = torch.tensor(x)
        y = self.get_y(pdb)
        edge_index = torch.tensor(edge_index).T
        edge_attr = torch.tensor(edge_attr) if edge_attr else None
        pos = torch.tensor(pos) if pos else None

        if not edge_index.shape[0]:
            raise ValueError('No edge in node.')

        data = torchg.data.Data(x, edge_index, edge_attr, y, pos)
        data.pdb = Path(pdb).name

        if data_attr_dict is not None:
            for attr_name, attr_func in data_attr_dict.items():
                setattr(data, attr_name, attr_func(pose, pdb))

        assert data.is_undirected(), f'Data {pdb.stem} is directed and is not supported.'
        return data

    def save_graph(self, pdb: AnyPath, path: AnyPath, shared_pose: bool = False,
                   data_attr_dict: Optional[Dict[str, Callable]] = None, **kwargs) -> bool:
        if not path.exists():
            sample_data = self.process_graph(pdb, shared_pose=shared_pose, data_attr_dict=data_attr_dict, **kwargs)
            torch.save(sample_data, path)

        return True

    def create_dataset(self, naming_func: Optional[Callable] = None, overwrite: bool = False,
                       shared_pose: bool = False, n_processes: int = 1,
                       data_attr_dict: Optional[Dict[str, Callable]] = None, **kwargs) -> None:
        """Iterate through, process and save all samples under predataset."""
        create_dir(self.dataset_path, overwrite)

        inputs = []
        for pdb in get_directory_pdb(src_dir=self.predataset_path, deep=True):
            if naming_func:
                path = naming_func(str(pdb))
            else:
                path = pdb.stem + '.pt'
            path = self.dataset_path / path

            if not path.exists():
                inputs.append((pdb, path))

        if shared_pose:  # assume all poses the same to speed up
            if n_processes == 1:
                self.set_shared(inputs[0][0])
            else:
                raise NotImplementedError('Sharing pose object not allowed by multiprocessing.')

        if n_processes == 1:
            for i, (pdb, path) in enumerate(inputs, 1):
                print(f'\rDataset processing [{i}|{len(inputs)}]'.ljust(75, ' '), end='')
                self.save_graph(pdb, path, shared_pose, data_attr_dict, **kwargs)

        elif n_processes > 1:
            pool = mlp.Pool(n_processes)
            pbar = tqdm(desc=self.name, total=len(inputs), ncols=75)

            for i, (pdb, path) in enumerate(inputs, 1):
                pool.apply_async(self.save_graph, args=(pdb, path, shared_pose, data_attr_dict), kwds=kwargs,
                                 callback=lambda x: pbar.update())

            pool.close()
            pool.join()


class BaseDatamodule(pl.LightningDataModule):
    def __init__(self, dataset: Any, batch_size: int = 1, train_val_test_ratio: Tuple[AnyNum, AnyNum, AnyNum] = (8, 1, 1),
                 num_workers: int = 1, pin_memory: bool = False, shuffle: bool = True, n_ensemble: int = 1,
                 split_root: Optional[AnyPath] = None):
        super().__init__()
        self.dataloader_kwargs = {'batch_size': batch_size, 'num_workers': num_workers, 'pin_memory': pin_memory,
                                  'shuffle': shuffle}
        self.train_val_test_ratio = (
            train_val_test_ratio[0] / sum(train_val_test_ratio),
            train_val_test_ratio[1] / sum(train_val_test_ratio),
            train_val_test_ratio[2] / sum(train_val_test_ratio),
        )
        self.dataset = dataset
        self.split_root = self.dataset.root if split_root is None else Path(split_root)

        self.n_ensemble = n_ensemble
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.setup()

        self._example_input_array = None

    @property
    def example_input_array(self):
        if self._example_input_array is None:
            self._example_input_array = next(iter(self.train_dataloader()))
        return self._example_input_array

    @property
    def batch_size(self):
        return self.dataloader_kwargs['batch_size']

    @batch_size.setter
    def batch_size(self, batch_size):
        self.dataloader_kwargs['batch_size'] = batch_size

    @property
    def num_workers(self):
        return self.dataloader_kwargs['n_workers']

    def get_n_samples(self, stage: Optional[str] = None):
        if stage is None:
            return len(self.dataset)
        if stage == 'train':
            return len(self.train_set)
        if stage in ('val', 'validation'):
            return len(self.val_set)
        if stage == 'test':
            return len(self.test_set)
        raise ValueError('Only support train, val, test stages.')

    def __len__(self):
        return self.get_n_samples()

    def setup(self, stage: Optional[str] = None):
        pt_train = self.split_root / 'train_indices.pt'
        pt_val = self.split_root / 'val_indices.pt'
        pt_test = self.split_root / 'test_indices.pt'

        if pt_train.exists() and pt_val.exists() and pt_test.exists():
            train_indices = torch.load(pt_train)
            val_indices = torch.load(pt_val)
            test_indices = torch.load(pt_test)
        else:
            n_samples = self.get_n_samples()

            if self.n_ensemble != 1:
                if n_samples % self.n_ensemble != 0:
                    raise ValueError(f'Number of processed files is not a multiple of n_ensemble '
                                     f'({n_samples}|{self.n_ensemble}).')
                n_samples = self.get_n_samples() // self.n_ensemble

            m = int(n_samples * self.train_val_test_ratio[0])
            n = int(n_samples * self.train_val_test_ratio[1])
            l = n_samples - m - n
            indices = np.random.permutation(np.arange(n_samples))
            train_indices = torch.tensor(indices[:m])
            val_indices = torch.tensor(indices[m:m+n])
            test_indices = torch.tensor(indices[m+n:m+n+l])

            if self.n_ensemble > 1:
                train_indices = train_indices * self.n_ensemble
                train_indices = torch.cat([train_indices + i for i in range(self.n_ensemble)])
                val_indices = val_indices * self.n_ensemble
                val_indices = torch.cat([val_indices + i for i in range(self.n_ensemble)])
                test_indices = test_indices * self.n_ensemble
                test_indices = torch.cat([test_indices + i for i in range(self.n_ensemble)])

            train_indices = train_indices.tolist()
            val_indices = val_indices.tolist()
            test_indices = test_indices.tolist()

            torch.save(train_indices, pt_train)
            torch.save(val_indices, pt_val)
            torch.save(test_indices, pt_test)

        self.train_set = torch.utils.data.Subset(self.dataset, train_indices)
        self.val_set = torch.utils.data.Subset(self.dataset, val_indices)
        self.test_set = torch.utils.data.Subset(self.dataset, test_indices)

    def train_dataloader(self) -> torchg.data.DataLoader:
        return torchg.loader.DataLoader(self.train_set, **self.dataloader_kwargs)

    def val_dataloader(self) -> torchg.data.DataLoader:
        kwargs = self.dataloader_kwargs.copy()
        kwargs['shuffle'] = False
        return torchg.loader.DataLoader(self.val_set, **kwargs)

    def test_dataloader(self) -> torchg.data.DataLoader:
        kwargs = self.dataloader_kwargs.copy()
        kwargs['shuffle'] = False
        return torchg.loader.DataLoader(self.test_set, **kwargs)


def main():
    pass


if __name__ == '__main__':
    main()
