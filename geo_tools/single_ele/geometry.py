import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from collections import Counter
import os


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

def equ_sites(positions,cutoff):
    """
    :param positions: coordinates
    :param cutoff:    cutoff distance defined by mutiple scattering radius
    :return:          non-equ position indexes
    """
    # cutoff method
    def duplicates(lst, item):
        """
        :param lst: the whole list
        :param item: item which you want to find the duplication in the list
        :return: the indexes of duplicate items
        """
        return [i for i, x in enumerate(lst) if x == item]

    dis_all =  np.around(distance_matrix(positions,positions),decimals=4)
    dis_all.sort(axis=1)
    dis_cut = [list(dis_all[i][dis_all[i] < cutoff]) for i in range(len(dis_all))]
    dup = []
    for i in range(len(dis_cut)):
        dup.append(duplicates(dis_cut, dis_cut[i])[0])
    #unique_index = list(set(dup))  # set can delete all duplicated items in a list
    unique_index = dict()
    for i in range(len(dup)):
        if dup[i] in unique_index:
            unique_index[dup[i]].append(i)
        else:
            unique_index.update({dup[i]:[i]})
    # sort it using sorted method. Do not use list.sort() method, because it returns a nonetype.
    #unique_index = np.array(sorted(unique_index))
    #print("number of atoms: {}".format(len(positions)))
    #print("number of unique atoms: {}".format(len(atom_index))) #
    return unique_index  #keys are those unique sites, values are the cooresponding equ-sites for those unique_sites

def getCN_dis_Oneshell(positions,centerindex,N):
    CN = []
    dis_all = np.around(distance_matrix([positions[centerindex]], positions), decimals=4) #must add [] here
    dis_all.sort(axis=1)
    freq = dict(Counter(list(dis_all[0])))
    distances = list(freq.keys())[N]
    CN = list(freq.values())[N]
    return CN,distances
def moment_descriptor(atom:Atoms):
    moment_atom=atom.get_moments_of_inertia(vectors=False)
    I=np.sort(moment_atom)
    zeta=((I[2]-I[1])**2+(I[1]-I[0])**2+(I[0]-I[2])**2)/(I[0]**2+I[1]**2+I[2]**2)
    eta=(2*I[1]-I[0]-I[2])/I[2]
    dis=distance_matrix(atom.arrays['positions'],atom.arrays['positions'])
    dis_sort=np.round(np.sort(dis,axis=1),5)
    cn_n=[]
    for i in range(len(dis_sort)):
        cn=np.unique(dis_sort[i],return_counts=True)[1][1]
        cn_n.append(cn)
    cn_n = np.array(cn_n)
    mean_c=np.mean(cn_n)
    RMS_c=np.sqrt(np.sum((cn_n-mean_c)**2/len(cn_n)))
    if eta<10e-10 and eta>-10e-10:
        eta=0.0
    if zeta<10e-10 and zeta>-10e-10:
        zeta=0.0
    return {"Departure from sphere":np.round(zeta,6),
            "oblateness":np.round(eta,6),
            "mean_c":np.round(mean_c,6),
            "RMS_c":np.round(RMS_c,6),
            "min_c":np.min(cn_n),
            "max_c":np.max(cn_n)}


