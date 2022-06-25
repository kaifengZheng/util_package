import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from collections import Counter
import os
from ase.io.vasp import read_vasp
import matplotlib.pyplot as plt
from ase.build import make_supercell
from scipy.spatial import distance_matrix
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import LocalGeometryFinder
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import MultiWeightsChemenvStrategy
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import LightStructureEnvironments
import pymatgen as mg


"""
supercell:
      CrCl3=read_vasp("POSCAR")
      P=[[3,0,0],[0,3,0],[0,0,1]]
      supercell=make_supercell(CrCl3, P, wrap=True, tol=1e-05)

"""

def radius(positions):
    """
    This algorithm calculates the radius of
    the circumscribed sphere of the polyhedron
    particle
    :param positions: coordinates
    :return: radius of particle
    """
    center = positions.mean(axis=0)
    vector = positions - center
    dis = np.linalg.norm(vector,axis=1)
    R = np.max(dis)
    return R


def my_ceil(a, precision=2):
    return np.round(a + 0.5 * 10 ** (-precision), precision)


def center_atom_finder(ele_num, supercell_info):
    positions = supercell_info.arrays['positions']
    num = supercell_info.arrays['numbers']
    center_pos = np.mean(positions, axis=0)
    atom_dis = distance_matrix([center_pos], positions)
    stacked_data = np.hstack((num.reshape(-1, 1), atom_dis.reshape(-1, 1), positions[:, :]))
    data = pd.DataFrame(stacked_data, columns=["atom", "dis2center", "x", "y", "z"])
    data_ele = data[data['atom'] == float(ele_num)]
    center_ele = data_ele[data_ele['dis2center'] == data_ele['dis2center'].min()]
    index = center_ele.index[0]

    center_atom_pos = np.array([center_ele['x'], center_ele['y'], center_ele['z']]).reshape(1, 3)
    update_atom_dis = distance_matrix(center_atom_pos, positions).reshape(-1, 1)
    data['dis2center'] = data['dis2center'].replace(np.array(data['dis2center']), update_atom_dis)

    # shift the center position to the assigned center atom
    data['x'] = data['x'] - center_atom_pos[0][0]
    data['y'] = data['y'] - center_atom_pos[0][1]
    data['z'] = data['z'] - center_atom_pos[0][2]
    data = data.sort_values(by="dis2center", ascending=True)
    return index, data


def nn_dis_rough(data, layer=1):
    return my_ceil(data['dis2center'], 2).unique()[layer]


def atom_by_layers(ele_num, supercell, layer=1):
    _, data = center_atom_finder(ele_num, supercell)
    distance = nn_dis_rough(data, layer)
    supercell_2 = supercell.copy()
    data_cut_sphere = data[data['x'] ** 2 + data['y'] ** 2 + data['z'] ** 2 <= distance ** 2]
    rebu_coord = np.hstack((np.array(data_cut_sphere['x']).reshape(-1, 1),
                            np.array(data_cut_sphere['y']).reshape(-1, 1),
                            np.array(data_cut_sphere['z']).reshape(-1, 1)))
    rebu_num = np.array(data_cut_sphere['atom'])
    supercell_2.arrays['positions'] = rebu_coord
    supercell_2.arrays['numbers'] = np.int0(rebu_num)
    print(
        f"#{layer} layer is {distance} Angstrom from center element (The distance is the 2 digits ceil of exact value of nn distance).")
    # view(supercell_2,viewer='x3d')=
    return supercell_2


