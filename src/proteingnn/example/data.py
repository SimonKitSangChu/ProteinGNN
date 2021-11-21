from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import math
from io import StringIO
from torch.utils.data import DataLoader

from ..data import *

AnyPath = Union[str, PosixPath, WindowsPath]
AnyNum = Union[int, float]


def _log2en_rcsb_mismatch(log2en: pd.DataFrame, seqrec_wt: Iterable) -> List[int]:
    """Find resid mismatch between log2en csv and rcsb fasta."""
    mismatch_resid = []
    for resid, (aa1, aa2) in enumerate(zip(seqrec_wt, log2en.wtAA), 1):
        if aa1 != aa2:
            mismatch_resid.append(resid)

    return mismatch_resid


def get_dms(seqrec_wt: Iterable, mutations: Iterable, sep: str = ':', debug=False) -> Dict[str, Iterable]:
    seq2 = str(seqrec_wt.seq)

    seqrec_dict = {}
    for mutation in mutations:
        seq = str(seqrec_wt.seq)

        for mut in mutation.split(sep):
            wt_aa, resid, mt_aa = mut[0], int(mut[1:-1]), mut[-1]

            if wt_aa != seqrec_wt[resid-1]:
                if debug:
                    if resid == len(seq2):
                        seq2 = seq2[:resid - 1] + mt_aa
                    else:
                        seq2 = seq2[:resid - 1] + mt_aa + seq2[resid:]
                else:
                    raise ValueError(f'{mutation} does not match sequence.')

            if resid == len(seq):
                seq = seq[:resid-1] + mt_aa
            else:
                seq = seq[:resid-1] + mt_aa + seq[resid:]

            assert seq[resid-1] == mt_aa
            assert len(seq) == len(seqrec_wt)

        seqrec_dict[mutation] = SeqRecord(seq, id=mutation)

    if seq2 != str(seqrec_wt.seq):
        raise ValueError(f'Sequence does not match mutation. The anticipated sequence is: {seq2}')

    return seqrec_dict


def read_DeepSequence_csv(csv, cols: Optional[List] = None, drop_wt: bool = True, verbose: bool = False) -> pd.DataFrame:
    csv = Path(csv)
    if cols is None:
        cols = []

    comments = []
    data = []

    col_str = '# Experimental data columns: '

    for line in csv.open('r'):
        line = line.replace('\n', '')
        if line[0] == '#':
            comments.append(line)
            if col_str in line and not cols:
                cols = line.replace(col_str, '').split(', ')
        else:
            data.append(line)

    data = StringIO('\n'.join(data))
    df = pd.read_csv(data, sep=';')
    df = df[['mutant'] + cols]

    if len(cols) != 1 and verbose:
        print(csv, cols, '\n')

    df = df.set_index('mutant')

    if ('wt' in df.index or 'WT' in df.index) and verbose:
        warnings.warn(f'Drop wt data from {csv}.')

    if drop_wt:
        index = [x for x in df.index if x not in ('wt', 'WT')]
        df = df.loc[index]

    return df


def bglb_Tm_pose2pdb(pdb : Optional[AnyPath] = None) -> Dict:
    """pose2pdb from Rosetta to rcsb resid mapping for 2jie"""
    if not globals()['PYROSETTA_INIT']:
        pyrosetta.init('-out::level 0')
        globals()['PYROSETTA_INIT'] = True

    if pdb is None:
        pdb = 'data/2jie.pdb'

    pose = pyrosetta.pose_from_pdb(str(pdb))

    pose2pdb = {}
    for rosid in range(1, len(pose.residues) + 1):
        resid = pose.pdb_info().pose2pdb(rosid)
        resid = int(resid.split()[0])
        pose2pdb[rosid] = resid

    return pose2pdb


def read_bglb(csv: Optional[AnyPath] = None, y_name: str = 'Tm', resid_mapping: Optional[Dict] = None):
    """(Legacy) Read bglb data from the sources
        (T50) DOI:10.1371/journal.pone.0147596
        (Tm) unpublished

    args:
         y_name: Experiment quantity allowed - 'Tm' or 'T50'
         resid_mapping: Subscribable to map resid to convert rosetta resid into pdb resid.
    """

    if y_name == 'Tm':
        if csv is None:
            csv = 'bglb/data/Final_tm_kinetic_1.csv'
        df = pd.read_csv(csv).rename(columns={'Mutant': 'mutant', 'tm': 'Tm'})
    elif y_name == 'T50':
        if csv is None:
            csv = 'bglb/data/tm_t50.csv'
        df = pd.read_csv(csv).rename(columns={'Protein': 'mutant', 'tm': 'Tm', 't50': 'T50'})
    else:
        raise ValueError('Only has Tm or T50 data.')

    df = df.set_index('mutant').dropna(subset=[y_name])
    exp_data = df[y_name].to_dict()
    exp_data = OrderedDict((mut_name, exp_data[mut_name]) for mut_name in sorted(exp_data))

    if resid_mapping is not None:
        def mut_name_map(mut_name):
            resid_from = int(mut_name[1:-1])
            resid_to = resid_mapping[resid_from]
            return mut_name[0] + str(resid_to) + mut_name[-1]

        exp_data = OrderedDict((mut_name_map(u), v) for u, v in exp_data.items())

    return exp_data


def read_log2en(csv: Optional[AnyPath] = None, return_type: str = 'DataFrame', keep_dummies: bool = False,
                skip_nan: bool = True):
    """(Legacy) skip_nan option only available for return_type OrderedDict."""

    if return_type not in ('DataFrame', 'Array', 'Tensor', 'OrderedDict'):
        raise ValueError('Only DataFrame, Array, Tensor and OrderedDict supported.')

    if csv is None:
        csv = 'data/beta_glu_DMS_log2enrichment.csv'
    else:
        csv = Path(csv)

    df = pd.read_csv(csv)
    df = df.rename(columns={'pos': 'resid'})
    df = df[df['resid'] < 480]
    df = df.set_index('resid')

    if not keep_dummies:
        for resid in df.index:
            wt_aa = df.loc[resid, 'wtAA']
            df.loc[resid, wt_aa] = float('nan')

    # throw away resid which has mismatch between log2en csv and rcsb fasta
    df.loc[10, :] = np.nan

    if return_type == 'DataFrame':
        return df
    elif return_type == 'Array':
        return df[AA_ALPHABETS].values
    elif return_type == 'Tensor':
        return torch.from_numpy(df[AA_ALPHABETS].values)
    else:
        log2en = OrderedDict()
        for resid in df.index:
            wt_aa = df.loc[resid, 'wtAA']
            for aa in AA_ALPHABETS:
                log2en[f'{wt_aa}{resid}{aa}'] = df.loc[resid, aa]

        if skip_nan:
            log2en = OrderedDict((u, v) for u, v in log2en.items() if not math.isnan(v))

        return log2en


class EmbLibraryCreator:
    def __init__(self, rootdir: AnyPath, fasta: AnyPath, exp_data: OrderedDict, pssm: Optional[AnyPath] = None):
        self.rootdir = Path(rootdir)
        if not self.rootdir.exists():
            self.rootdir.mkdir()

        self.seqrec_wt = SeqIO.read(fasta.open('r'), format='fasta')
        if pssm is None:
            self.pssm2D = None
        else:
            self.pssm2D = pssm2D(return_type='Tensor', relative=False)
        self.exp_data = exp_data

    def create_mutant_pdbs(self, pdb_code: str, src_dir: AnyPath, dst_dir: AnyPath, pdb: Optional[str] = None):
        """Create (dummy) mutant pdb for dataset generation.

        Assume input pdb(s) as example/data/{pdb_code}.pdb
        Note that the mutant pdbs are dummies, i.e. only the name has changed.
        """
        if pdb is None:
            pdb = f'{pdb_code}.pdb'

        src_dir = Path(src_dir).absolute()
        if not src_dir.exists():
            raise FileNotFoundError(f'{src_dir} not found.')

        dst_dir = Path(dst_dir).absolute()
        if not dst_dir.exists():
            dst_dir.mkdir()

        # print('Start mutant pdb generation.')
        for mut_name, seqrec in get_dms(self.seqrec_wt, mutations=self.exp_data).items():
            # print(f'Generate mutant \r{mut_name}'.ljust(40, ' '), end='')
            src_pdb = src_dir / pdb
            mt_pdb = dst_dir / f'{pdb_code}_{mut_name}.pdb'
            if not mt_pdb.exists():
                mt_pdb.symlink_to(src_pdb)

    def create_embedding_library(self, embedding_name: str, pdb_code: str, pssm_dim: Optional[int] = None,
                                 use_diff: bool = False, pdb_code_ref: Optional[str] = None, **kwargs):
        embedding_dir = self.get_embedding_dir(embedding_name, pssm_dim)
        pure_embedding_dir = self.get_embedding_dir(embedding_name, pssm_dim=None)
        if not embedding_dir.exists():
            embedding_dir.mkdir()

        if pdb_code_ref:
            src_pts = list(embedding_dir.glob(f'{pdb_code_ref}*.pt'))
            if not src_pts:
                raise FileNotFoundError(f'No {pdb_code_ref}*.pt found.')

            for src_pt in src_pts:
                dst_pt = str(src_pt).replace(pdb_code_ref, pdb_code)
                dst_pt = Path(dst_pt)
                if not dst_pt.exists():
                    dst_pt.symlink_to(src_pt.name)

            return True

        if use_diff:
            pt = embedding_dir / f'{pdb_code}_wildtype.pt'
            pure_pt = pure_embedding_dir / pt.name
            if pt.exists():
                embedding_wt = torch.load(pt)
            else:
                pure_embedding = torch.load(pure_pt) if pure_pt.exists() else None
                embedding_wt = self.get_embedding(embedding_name, self.seqrec_wt, pssm_dim, pure_embedding, **kwargs)
                torch.save(embedding_wt, pt)
        else:
            embedding_wt = None

        print(f'Start {embedding_dir.stem} library creation.')
        dms = get_dms(self.seqrec_wt, self.exp_data)

        for i, (mut_name, seqrec) in enumerate(dms.items(), 1):
            print(f'\r[{i}|{len(dms)}]    {mut_name}'.ljust(50, ' '), end='')

            pt = embedding_dir / f'{pdb_code}_{mut_name}.pt'
            pure_pt = pure_embedding_dir / pt.name

            if not pt.exists():
                if pure_pt.exists():
                    pure_embedding = torch.load(pure_pt)
                else:
                    pure_embedding = None

                embedding = self.get_embedding(embedding_name, seqrec, pssm_dim, embedding_wt, pure_embedding, **kwargs)
                torch.save(embedding, pt)

        print(f'\rComplete library creation.')

    def get_embedding_dir(self, embedding_name, pssm_dim):
        if embedding_name == 'pssm':
            if pssm_dim is None:
                return self.rootdir / 'pssm'
            else:
                return self.rootdir / f'{pssm_dim}Dpssm'

        if pssm_dim is None:
            return self.rootdir / f'{embedding_name}'
        else:
            return self.rootdir / f'{embedding_name}{pssm_dim}Dpssm'

    def get_embedding(self, embedding_name, *args, **kwargs):
        method_name = f'{embedding_name}_embedding'
        try:
            return getattr(self, method_name)(*args, **kwargs).to(dtype=torch.float)
        except AttributeError:
            raise NotImplementedError(f'{embedding_name} not implemented nor found in creator.')

    def pssm_embedding(self, seqrec, pssm_dim, embedding_wt=None, pure_embedding=None):
        if pssm_dim == 1:
            return pssm1D(seq=seqrec, return_type='Tensor', relative=True).to(dtype=torch.float)
        elif pssm_dim == 2:
            # relative pssm to sequence
            return pssm1D(seq=seqrec, return_type='Tensor', relative=False).to(dtype=torch.float)\
                   - self.pssm2D.to(dtype=torch.float)
        else:
            raise ValueError('Invalid pssm_dim.')

    def esm_embedding(self, seqrec, pssm_dim, embedding_wt=None, pure_embedding=None, **kwargs):
        data = [('dummy', seqrec)]
        if pure_embedding is None:
            embedding = get_esm_representations(data, **kwargs)
            if embedding_wt is not None:
                embedding = embedding - embedding_wt
        else:
            embedding = pure_embedding

        if pssm_dim is None:
            return embedding
        elif pssm_dim == 1:
            pssm_values = pssm1D(seqrec, return_type='Tensor')
            # noinspection PyArgumentList
            return torch.cat([embedding, pssm_values], axis=1)
        elif pssm_dim == 2:
            # noinspection PyArgumentList
            return torch.cat([embedding, self.pssm2D], axis=1)
        else:
            raise ValueError('Invalid pssm_dim.')

    def onehot_embedding(self, seqrec, pssm_dim, embedding_wt=None, pure_embedding=None):
        if pure_embedding is None:
            embedding = seq2onehot(seqrec)
        else:
            embedding = pure_embedding

        if pssm_dim is None:
            return embedding.to(dtype=torch.float)
        elif pssm_dim == 1:
            pssm = pssm1D(seqrec, return_type='Tensor')
            # noinspection PyArgumentList
            return torch.cat((embedding, pssm), axis=1).type(dtype=torch.float)
        elif pssm_dim == 2:
            # noinspection PyArgumentList
            return torch.cat((embedding, self.pssm2D), axis=1).type(dtype=torch.float)
        else:
            raise ValueError('Invalid pssm_dim.')


class DefaultDataset(torchg.data.Dataset):
    def __init__(self, root: AnyPath, exp_data: Union[Dict, OrderedDict],
                 transform: Optional[callable] = None, prediction_type: str = 'regression'):
        if prediction_type == 'regression':
            transform = None
        elif prediction_type == 'classification':
            def binning(data, *args):
                data.y = ...
            transform = binning

        super().__init__(Path(root), transform)
        self.exp_data = exp_data

        self.root = Path(root)
        self.mut_names = list(self.exp_data.keys())
        self.y = list(self.exp_data.values())  # TODO binning for classification

        sample_pts = {}
        for filename in self.processed_file_names:
            key = '_'.join(reversed(Path(filename).stem.split('_')))
            sample_pts[key] = filename
        self.sample_pts = [sample_pts[key] for key in sorted(sample_pts)]

    @property
    def example_input_array(self):
        return self[0][0].clone()

    @property
    def num_edge_features(self):
        return self.example_input_array.num_edge_features

    @property
    def num_features(self) -> int:
        return self.example_input_array.num_features

    @property
    def num_node_features(self) -> int:
        return self.example_input_array.num_features

    @property
    def num_nodes(self) -> int:
        return self.example_input_array.num_nodes

    @property
    def num_edges(self) -> int:
        return self.example_input_array.num_edges

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return [x for x in self.root.glob('*.pt') if 'indices' not in x.name]

    def idx2x(self, idx, **kwargs):
        x = torch.load(self.sample_pts[idx], **kwargs)
        return x

    def __len__(self):
        return len(self.sample_pts)

    def __getitem__(self, idx: Union[int, slice], **kwargs):
        if type(idx) is int:
            x = self.idx2x(idx, **kwargs)
        elif type(idx) == slice:
            x = [self.idx2x(i, **kwargs) for i in np.arange(len(self))[idx]]
            x = torchg.data.Batch.from_data_list(x)
        else:
            raise TypeError('idx must be int or slice.')
        return x


class SingleSiteDataset(torch.utils.data.Dataset):
    def __init__(self, root: AnyPath, exp_data: Union[Dict, OrderedDict], pdb_code: str,
                 prediction_type: str = 'repression'):
        """Romero dataset of embeddings AT THE MUTATION SITE.

        args:
            root: Directory hosting the embeddings (or PSSM).
            exp_data: Key must be the mutation name, such as A123C. Value is the experiment value.
        """
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(f'Dataset {root} not found.')
        self.root = root

        super().__init__()
        self.exp_data = exp_data.copy()

        self.pdb_code = pdb_code
        self.mut_names = list(self.exp_data.keys())
        self.y = list(self.exp_data.values())

        if prediction_type == 'classification':  # TODO
            raise NotImplementedError

    @property
    def example_input_array(self):
        return self[0][0].clone()

    def idx2pt(self, idx):
        return self.root / f'{self.pdb_code}_{self.mut_names[idx]}.pt'

    def idx2x(self, idx, **kwargs):
        pt = self.root / f'{self.pdb_code}_{self.mut_names[idx]}.pt'
        resid = int(self.mut_names[idx][1:-1])
        x = torch.load(pt, **kwargs)
        return x[resid-1]

    def __len__(self):
        return len(self.mut_names)

    def __getitem__(self, idx: Union[int, slice, Iterable], **kwargs):
        if type(idx) is int:
            x = self.idx2x(idx, **kwargs)
            y = torch.tensor(self.exp_data[self.mut_names[idx]])
        elif type(idx) == slice:
            indices = np.arange(len(self))[idx]
            x = [self.idx2x(i, **kwargs) for i in indices]
            x = torch.cat(x, axis=0)
            y = torch.tensor([self.exp_data[self.mut_names[i]] for i in indices])
        elif type(idx) == Iterable:
            x = [self.idx2x(i, **kwargs) for i in idx]
            x = torch.cat(x, axis=0)
            y = torch.tensor([self.exp_data[self.mut_names[i]] for i in idx])
        else:
            raise TypeError('idx must be int or slice.')
        return x.to(dtype=torch.float), y.to(dtype=torch.float)


class DefaultDatamodule(BaseDatamodule):
    def __init__(self, batch_size: int = 1, train_val_test_ratio: Tuple[AnyNum, AnyNum, AnyNum] = (8, 1, 1),
                 num_workers: int = 1, pin_memory: bool = False, shuffle: bool = True, n_ensemble: int = 1,
                 split_root: Optional[AnyPath] = None, **kwargs):
        dataset = DefaultDataset(**kwargs)
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            train_val_test_ratio=train_val_test_ratio,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
            n_ensemble=n_ensemble,
            split_root=split_root
        )

    def print_summary(self):
        print(f'Dataset: {self}:')
        print('====================')
        print(f'Number of graphs: {len(self)}')
        print(f'Number of nodes: {self.num_nodes}')
        print(f'Number of edges: {self.num_edges}')
        print(f'Number of node features: {self.num_node_features}')
        print(f'Number of edge features: {self.num_edge_features}')
        print('====================')
        print(f'Average node degree: {self.num_edges / self.num_nodes:.2f}')
        print(f'Contains isolated nodes: {self.example_input_array.contains_isolated_nodes()}')
        print(f'Contains self-loops: {self.example_input_array.contains_self_loops()}')
        print(f'Is undirected: {self.example_input_array.is_undirected()}')
        print('====================')

    @property
    def log2en(self):
        return self.dataset.exp_data

    @property
    def mut_names(self):
        return self.dataset.seq_names

    @property
    def num_edge_features(self):
        return self.example_input_array[0].num_edge_features

    @property
    def num_node_features(self):
        return self.example_input_array[0].num_node_features

    @property
    def num_nodes(self) -> int:
        return self.example_input_array[0].num_nodes

    @property
    def num_edges(self) -> int:
        return self.example_input_array[0].num_edges
