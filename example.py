from pathlib import Path
import sys

from proteingnn.data import *


def main():
    pyrosetta.init()

    fa_factory = DatasetFactory(name='FullAtomDatasetFactory')

    fa_factory.node_filter = BaseNodeFilter()
    fa_factory.node_featurizer = AtomtypeNodeFeaturizer()
    # fa_factory.edge_featurizer = BondedEdgeFeaturizer()
    # fa_factory.edge_featurizer = HbondEdgeFeaturizer(is_edge_only=False)

    featurizers = [
        BondedEdgeFeaturizer(is_edge_only=False),
        DistanceEdgeFeaturizer(max_distance=3, is_edge_only=False),
        HbondEdgeFeaturizer(is_edge_only=False),
    ]
    edge_featurizer = CompositeEdgeFeaturizer(
        name='CompositeEdgeFeaturizer',
        featurizers=featurizers,
        all_is_edge=False,
    )
    fa_factory.edge_featurizer = edge_featurizer

    # test on single datum
    pdbs = get_directory_pdb(fa_factory.predataset_path, deep=True)
    pdb = next(pdbs)
    data = fa_factory.process_complex(pdb)

    # test on full data
    # fa_factory.create_dataset(pos_flag=True, n_processes=1)


if __name__ == '__main__':
    main()
