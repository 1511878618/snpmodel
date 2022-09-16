# -*- coding: utf-8 -*-

import os

import pandas as pd
import torch
from pyuul import VolumeMaker  # the main PyUUL module
from pyuul import utils  # the PyUUL utility module

"""
常用的函数集合
    - Accumulator
    - changeStrByPos
"""


class Accumulator:
    """ 在n个变量上累加 """
    def __init__(self, n):
        self.data = [0.0] * n 
    
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    
    def reset(self):
        self.data = [0.0] * len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
    

def changeStrByPos(string:str, idx, new):
    """
    更改一个字符串的某个位置的元素
    """
    tmp = list(string)
    tmp[idx] = str(new)
    return ''.join(tmp)
    

def modelParametersNum(model):
    totalNum = sum([i.numel() for i in model.parameters()])
    print(f'模型总参数个数：{sum([i.numel() for i in model.parameters()])}')
    return totalNum
    
def mkdirs(path):
    try:
        os.makedirs(path)
    except:
        pass

def try_gpu():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    return device




def splitAndConcatByRows(df, colName, pattern, reset_index=True):
    """
    splitAndConcatByRows pd.melt?

    Args:
        df (_type_): _description_
        colName (_type_): _description_
        pattern (_type_): _description_
        reset_index (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: 对每一行进行拆分，如果存在某列含有pattern，则对pattern进行split，得到的多个结果和该行其他列组成新的几行，并concat在一起
    """
    out = [] 
    for idx, series in df.iterrows():
        for i in str(series[colName]).split(pattern):
            series[colName] = i
            out.append(series)

    return pd.concat(out, axis=1).T.reset_index(drop=reset_index)



def searchList(x:list, pattern):
    """
    searchList 在list中搜索某个元素，返回对应的索引，没有则返回None

    Args:
        x (list): _description_
        pattern (any): 元素

    Returns:
        _type_: int or None 
    """

    for idx, element in enumerate(x):
        if pattern == element:
            return idx
    return None 


from multiprocessing import Pool


def multiprocessing_map(func, iter_list, processes=5):
    pool = Pool(processes)
    res = pool.map(func, iter_list)
    return res 

def get_VoxelRepresentation(pdb_path):
    coords, atname = utils.parsePDB(pdb_path) # get coordinates and atom names
    atoms_channel = utils.atomlistToChannels(atname) # calculates the corresponding channel of each atom
    radius = utils.atomlistToRadius(atname) # calculates the radius of each atom

    VoxelsObject = VolumeMaker.Voxels(sparse=True)

    coords = coords
    radius = radius
    atoms_channel = atoms_channel

    VoxelRepresentation = VoxelsObject(coords, radius, atoms_channel)
    return VoxelRepresentation


## load uniprot variants

def find_comment(file_path = "data/uniprot/variants/homo_sapiens_variation.txt", comment="#"):
    idx = 1
    for i in iter(open(file_path)):
        if "#" in i :
            return idx 
        else:
            idx+=1
            
def load_uniprot_variants(path = "data/uniprot/variants/homo_sapiens_variation.txt", **kwargs):
    data = pd.read_csv(path, sep="\t", skiprows=find_comment(path, comment="#"), comment="_", **kwargs)
    return data 



## 从PDB获取二级结构

## 需要安装dssp
#  conda install Biopython 
#  conda install -c salilab dssp

from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP 


def sec_structure_fromPDB(pdb):
    #  解析PDBfile
    p = PDBParser()
    structure = p.get_structure(pdb.split(".")[0], pdb)
    #  获取PDBfile中第一个结构，如果有多个的时候取第一个
    model = structure[0]
    dssp = DSSP(model, pdb, dssp='mkdssp')
    #  获得二级结构序列
    sequence = ""
    sec_structure = ""

    for property in dssp.property_list:
        sequence += property[1]
        sec_structure += property[2]
    return sequence, sec_structure

def save_sec_structure(raw_sequence, sec_structure, filename):
    with open(filename, "w") as f:
        f.write(raw_sequence + "\n")
        f.write(sec_structure)
    print("saveing to filename")