from typing import Dict
from pathlib import Path
import pyrosetta
import pytest
import sys

from proteingnn.data import pdb2pose_atoms, atom_id2atom_name,\
    BaseNodeFilter, ChainNodeFilter, AtomNameNodeFilter, CompositeNodeFilter, \
    BaseNodeFeaturizer, AtomtypeNodeFeaturizer, \
    BaseEdgeFeaturizer


pyrosetta.init('-mute all')
pose, atoms = pdb2pose_atoms('1gnx_chA.pdb')
atom_ids = sorted(atoms, key=lambda x: str(x))
atom_id0 = atom_ids[0]
atom_id1 = atom_ids[1]
# pose_chB, atoms_chB = pdb2pose_atoms('test_chB.pdb')
# pose, atoms = pdb2pose_atoms('test.pdb')


def _share_all_keys(dict1: Dict, dict2: Dict) -> bool:
    return all(key in dict1 for key in dict2) and all(key in dict2 for key in dict1)


class TestNodeFilters:
    def test_BaseNodeFilter(self):
        filter = BaseNodeFilter()
        new_atoms = filter.filter(pose, atoms)
        assert _share_all_keys(new_atoms, atoms)

    def test_AtomNameNodeFilter(self):
        filter = AtomNameNodeFilter(atom_name_pass=['C'])
        new_atoms = filter.filter(pose, atoms)
        assert len(new_atoms) == 447

        filter = AtomNameNodeFilter(atom_name_pass=['CA'])
        new_atoms = filter.filter(pose, atoms)
        assert len(new_atoms) == 447

        filter = AtomNameNodeFilter(atom_name_pass=['CB'])
        new_atoms = filter.filter(pose, atoms)
        assert len(new_atoms) == 408

    def test_CompositeNodeFilter(self):
        filter1 = BaseNodeFilter()
        filter2 = AtomNameNodeFilter(atom_name_pass=['C'])
        atoms1 = filter1.filter(pose, atoms)
        atoms2 = filter2.filter(pose, atoms)

        try:
            filter3 = CompositeNodeFilter(components=[])
        except ValueError:
            pass
        else:
            raise AssertionError('CompositeNodeFilter should not accept empty components.')

        try:
            filter3 = CompositeNodeFilter(components=BaseNodeFilter())
        except TypeError:
            pass
        else:
            raise AssertionError('CompositeNodeFilter should not accept single BaseNodeFilter.')

        filter3 = CompositeNodeFilter(components=[filter1, filter2], intersection=False)
        atoms3 = filter3.filter(pose, atoms)
        assert _share_all_keys(atoms1, atoms3)
        assert all(atom_id in atoms3 for atom_id in atoms2)

        filter3 = AtomNameNodeFilter(atom_name_pass=['CA'])
        filter4 = CompositeNodeFilter(components=[filter2, filter3], intersection=True)
        atoms4 = filter4.filter(pose, atoms)
        assert not atoms4

        filter4 = CompositeNodeFilter(components=[filter2, filter3], intersection=False)
        atoms4 = filter4.filter(pose, atoms)
        assert len(atoms4) == (447 + 447)


class TestNodeFeaturizer:
    def test_BaseNodeFeaturizer(self):
        featurizer = BaseNodeFeaturizer()
        try:
            featurizer.featurize(pose, atom_id0)
        except NotImplementedError:
            pass
        else:
            raise AssertionError('BaseNodeFeaturizer not returning NotImplementedError.')

    def test_AtomtypeNodeFeaturizer(self):
        atom_cats = {'C': 1, 'H': 2, 'O': 3, 'N': 4, 'S': 5}
        atomtypes = {
            'N': ['N', 'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NV', 'NZ'],
            'S': ['SD', 'SG'],
            'O': ['O', 'OD1', 'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH', 'OXT'],
            'C': ['C', 'CA', 'CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3', 'CG', 'CG1', 'CG2', 'CH2', 'CZ', 'CZ2', 'CZ3'],
            'H': ['1H', '1HA', '1HB', '1HD', '1HD1', '1HD2', '1HE', '1HE2', '1HG', '1HG1', '1HG2', '1HH1', '1HH2',
                  '1HZ', '2H', '2HA', '2HB', '2HD', '2HD1', '2HD2', '2HE', '2HE2', '2HG', '2HG1', '2HG2', '2HH1',
                  '2HH2', '2HZ', '3H', '3HB', '3HD1', '3HD2', '3HE', '3HG1', '3HG2', '3HZ', 'H', 'HA', 'HB', 'HD1',
                  'HD2', 'HE', 'HE1', 'HE2', 'HE3', 'HG', 'HG1', 'HH', 'HH2', 'HZ', 'HZ2', 'HZ3'],
        }
        featurizer = AtomtypeNodeFeaturizer(atomtypes=atomtypes, atom_cats=atom_cats)

        for atom_id in atoms:
            try:
                featurizer.featurize(pose, atom_id)
            except Exception as E:
                # raise E('AtomtypeNodeFeaturizer fails on 1gnx_chA.pdb.')
                print(atom_id2atom_name(atom_id, pose))

        answers = {
            1: atom_cats['N'],
            2: atom_cats['C'],
            3: atom_cats['C'],
            4: atom_cats['O'],
            5: atom_cats['C'],
        }

        for i, answer in answers.items():
            atom_id = pyrosetta.rosetta.core.id.AtomID(i, 1)
            encoding = featurizer.featurize(pose, atom_id)
            assert answer == (encoding.index(1) + 1)

        try:
            featurizer.atom_cats = {'C': 5, 'H': 2, 'O': 3, 'N': 4, 'S': 5}
        except ValueError:
            pass
        else:
            raise AssertionError('Failed at duplicate atom_cats indices.')

        try:
            featurizer.atom_cats = {'C': 1}
        except KeyError:
            pass
        else:
            raise AssertionError('Failed at missing atom_cats indices.')

    # TODO SeqEmbNodeFeaturizer
    # TODO CompositeNodeFeaturizer (pending for multiple NodeFeaturizers)


# class TestEdgeFeaturizer:
#     def test_BaseEdgeFeaturizer(self):
#         featurizer = BaseEdgeFeaturizer(is_edge_only=True)
#         try:
#             featurizer.featurize(pose, atom_id0, atom_id1)
#         except NotImplementedError:
#             pass
#         else:
#             raise AssertionError('Abstract method in BaseEdgeFeaturizer.')
#
#     def test_BondedEdgeFeaturizer(self):
#
#     # def test_DistanceEdgeFeaturizer(self):
#
#     # def test_HbondEdgeFeaturizer(self):
#
#     # def test_CompositeEdgeFeaturizer(self):
#
#
# def test_DatasetFactory():
#
