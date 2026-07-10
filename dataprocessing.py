# Parts of the dataset loading were adapted from the MolHFCNet repository:
# https://github.com/ndlongvn/MolHFCNet


import pandas as pd
import numpy as np
import os
import random
import json, pickle
#from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles, rdmolops
import networkx as nx
from Bio import SeqIO
from rdkit.Chem import AllChem, Descriptors
from utils import *


from rdkit import RDLogger
RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.info")
RDLogger.DisableLog("rdApp.error")   # optional

# feature dim
ATOM_DIM = 101
BOND_DIM = 12


ALLOWABLE_BOND_FEATURES = {
    'bond_type': ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'],
    'conjugated': ['T/F'],
    'stereo': ['STEREONONE', 'STEREOZ', 'STEREOE', 'STEREOCIS', 'STEREOTRANS', 'STEREOANY']
}


# one-hot encoding
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        # print(x)
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    '''Maps inputs not in the allowable set to the last element.'''
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def encoding_unk(x, allowable_set):
    list = [False for i in range(len(allowable_set))]
    i = 0
    for atom in x:
        if atom in allowable_set:
            list[allowable_set.index(atom)] = True
            i += 1
    if i != len(x):
        list[-1] = True
    return list
   
def get_atom_feature(atom):
    return np.array(
        one_of_k_encoding_unk(atom.GetSymbol(), [
            'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B',
            'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu',
            'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'
        ]) +
        one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        one_of_k_encoding_unk(atom.GetTotalValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        one_of_k_encoding_unk(atom.GetFormalCharge(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        [atom.GetIsAromatic()] +
        [atom.IsInRing()]
    )


#def get_bond_feature(bond):
#    return np.array(
 #       one_of_k_encoding(str(bond.GetBondType()), ALLOWABLE_BOND_FEATURES['bond_type']) +
 #       [bond.GetIsConjugated()] +
 #       one_of_k_encoding(str(bond.GetStereo()), ALLOWABLE_BOND_FEATURES['stereo'])
 #   )

def get_bond_feature(bond):
    return np.array(
        one_of_k_encoding(str(bond.GetBondType()), ALLOWABLE_BOND_FEATURES['bond_type']) +
        [bond.GetIsConjugated()] +
        [bond.IsInRing()] +
        one_of_k_encoding(str(bond.GetStereo()), ALLOWABLE_BOND_FEATURES['stereo'])
    )
    

#generate molecular graph -atom level

def mol_to_graphs(mol):
    atom_features, bond_list, bond_features = [], [], []

    # atom features
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_feature(atom).tolist())

    # bond features
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()

        bond_list.extend([[a1, a2], [a2, a1]])
        bond_features.extend([get_bond_feature(bond).tolist()] * 2)

    return atom_features, bond_list, bond_features

def load_ESOL_dataset(dataset):
    
    dataset_path = f'data/{dataset}/'
    
    input_df = pd.read_csv(dataset_path + f"refined_{dataset}.csv", sep=',')
    
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['measured']
    
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    
    return smiles_list, rdkit_mol_objs_list, labels.values



def load_BBBP_dataset(dataset):
    dataset_path = f"data/{dataset}/"
    input_df = pd.read_csv(dataset_path + f"refined_{dataset}.csv", sep=",")

    smiles_raw = input_df["SMILES"].astype(str).tolist()
    labels_raw = input_df["class"].copy()
    labels_raw = labels_raw.replace(0, -1)

    rdkit_mols = []
    smiles_ok = []
    labels_ok = []
    bad_idx = []

    for i, (s, y) in enumerate(zip(smiles_raw, labels_raw.values)):
        m = AllChem.MolFromSmiles(s)
        if m is None:
            bad_idx.append(i)
            continue
        # optional: canonicalize
        s_can = AllChem.MolToSmiles(m)
        rdkit_mols.append(m)
        smiles_ok.append(s_can)
        labels_ok.append(y)

    return smiles_ok, rdkit_mols, np.asarray(labels_ok)

def load_HIV_dataset(dataset):
    dataset_path = f"data/{dataset}/"
    input_df = pd.read_csv(dataset_path + f"refined_{dataset}.csv", sep=",")

    smiles_raw = input_df["SMILES"].astype(str).tolist()
    labels_raw = input_df["class"].copy()
    labels_raw = labels_raw.replace(0, -1)   # keep your {-1,1} convention

    mols = []
    smiles_ok = []
    labels_ok = []
    bad_idx = []

    for i, (s, y) in enumerate(zip(smiles_raw, labels_raw.values)):
        m = Chem.MolFromSmiles(s)
        if m is None:
            bad_idx.append(i)
            continue

        # optional but nice: canonicalize SMILES using the parsed mol
        s_can = Chem.MolToSmiles(m)
        mols.append(m)
        smiles_ok.append(s_can)
        labels_ok.append(y)

    return np.array(smiles_ok, dtype=object), mols, np.array(labels_ok)
    
def load_BACE_dataset(dataset):

    dataset_path = f'data/{dataset}/'
    
    input_df = pd.read_csv(dataset_path + f"refined_{dataset}.csv", sep=',')
  #  input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['SMILES']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['class']

    labels = labels.replace(0, -1)
   
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)

    return smiles_list, rdkit_mol_objs_list, labels.values

def load_ClinTox_dataset(dataset):

    dataset_path = f"data/{dataset}/"
    input_df = pd.read_csv(dataset_path + f"refined_{dataset}.csv", sep=",")

    tasks = ['FDA_APPROVED', 'CT_TOX']

    preprocessed_smiles_list = []
    preprocessed_rdkit_mol_objs_list = []
    labels_list = []

    for idx, row in input_df.iterrows():
        smiles = str(row['SMILES']).strip()

        if not smiles or smiles.lower() == 'nan':
            continue

        mol = AllChem.MolFromSmiles(smiles)
        if mol is None:
            continue

        canon_smiles = AllChem.MolToSmiles(mol)

        preprocessed_smiles_list.append(canon_smiles)
        preprocessed_rdkit_mol_objs_list.append(mol)
        labels_list.append(row[tasks].values)

    labels = pd.DataFrame(labels_list, columns=tasks)
    labels = labels.replace(0, -1)

    assert len(preprocessed_smiles_list) == len(preprocessed_rdkit_mol_objs_list)
    assert len(preprocessed_smiles_list) == len(labels)

    return preprocessed_smiles_list, preprocessed_rdkit_mol_objs_list, labels.values

def load_Tox21_dataset(dataset):

    dataset_path = f"data/{dataset}/"
    input_df = pd.read_csv(dataset_path + f"refined_{dataset}.csv", sep=",")

    tasks = [
        'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
        'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
    ]

    valid_smiles_list = []
    valid_mol_list = []
    valid_labels = []

    for idx, row in input_df.iterrows():
        smiles = str(row['SMILES']).strip()

        if not smiles or smiles.lower() == 'nan':
            continue

        mol = AllChem.MolFromSmiles(smiles)
        if mol is None:
            continue

        canon_smiles = AllChem.MolToSmiles(mol)

        valid_smiles_list.append(canon_smiles)
        valid_mol_list.append(mol)
        valid_labels.append(row[tasks].values)

    labels = pd.DataFrame(valid_labels, columns=tasks)

    # convert valid class label 0 -> -1, keep NaN as NaN
    labels = labels.replace(0, -1)

    assert len(valid_smiles_list) == len(valid_mol_list)
    assert len(valid_smiles_list) == len(labels)

    return valid_smiles_list, valid_mol_list, labels.values

def load_SIDER_dataset(dataset):
    """
    Returns:
        valid_smiles_list,
        valid_mol_list,
        labels.values
    """
    dataset_path = f"data/{dataset}/"
    input_df = pd.read_csv(dataset_path + f"refined_{dataset}.csv", sep=",")

    tasks = [
        'Hepatobiliary disorders',
        'Metabolism and nutrition disorders', 'Product issues', 'Eye disorders',
        'Investigations', 'Musculoskeletal and connective tissue disorders',
        'Gastrointestinal disorders', 'Social circumstances',
        'Immune system disorders', 'Reproductive system and breast disorders',
        'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
        'General disorders and administration site conditions',
        'Endocrine disorders', 'Surgical and medical procedures',
        'Vascular disorders', 'Blood and lymphatic system disorders',
        'Skin and subcutaneous tissue disorders',
        'Congenital, familial and genetic disorders',
        'Infections and infestations',
        'Respiratory, thoracic and mediastinal disorders',
        'Psychiatric disorders', 'Renal and urinary disorders',
        'Pregnancy, puerperium and perinatal conditions',
        'Ear and labyrinth disorders', 'Cardiac disorders',
        'Nervous system disorders',
        'Injury, poisoning and procedural complications'
    ]

    valid_smiles_list = []
    valid_mol_list = []
    valid_labels = []

    for idx, row in input_df.iterrows():
        smiles = str(row['SMILES']).strip()

        if not smiles or smiles.lower() == 'nan':
            print(f"Skipping empty/NaN SMILES at row {idx}")
            continue

        mol = AllChem.MolFromSmiles(smiles)
        if mol is None:
            continue

        canon_smiles = AllChem.MolToSmiles(mol)

        valid_smiles_list.append(canon_smiles)
        valid_mol_list.append(mol)
        valid_labels.append(row[tasks].values)

    labels = pd.DataFrame(valid_labels, columns=tasks)

    # convert 0 to -1
    labels = labels.replace(0, -1)

    assert len(valid_smiles_list) == len(valid_mol_list)
    assert len(valid_smiles_list) == len(labels)

    return valid_smiles_list, valid_mol_list, labels.values

def load_FreeSolv_dataset(dataset):

    dataset_path = f'data/{dataset}/'
    
    input_df = pd.read_csv(dataset_path + f"refined_{dataset}.csv", sep=',')
    
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['measured']
    
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


def load_Lipophilicity_dataset(dataset):
    """
    Regression task (Lipophilicity).
    Returns:
      smiles_ok (np.array of canonical smiles),
      mols (list of RDKit mol),
      labels_ok (np.array float)
    """
    dataset_path = f"data/{dataset}/"
    input_df = pd.read_csv(dataset_path + f"refined_{dataset}.csv", sep=",")

    smiles_raw = input_df["smiles"].astype(str).tolist()
    labels_raw = input_df["measured"].values.astype(np.float32)

    mols = []
    smiles_ok = []
    labels_ok = []
    bad_idx = []

    for i, (s, y) in enumerate(zip(smiles_raw, labels_raw)):
        m = AllChem.MolFromSmiles(s)
        if m is None:
            bad_idx.append(i)
            continue

        # optional: canonicalize smiles
        s_can = AllChem.MolToSmiles(m)
        mols.append(m)
        smiles_ok.append(s_can)
        labels_ok.append(float(y))

    return np.array(smiles_ok, dtype=object), mols, np.array(labels_ok, dtype=np.float32)
   
                               
def create_dataset(dataset):

    loader_map = {
        "BBBP": load_BBBP_dataset,
        "BACE": load_BACE_dataset,
        "HIV": load_HIV_dataset,
        "ClinTox": load_ClinTox_dataset,
        "Tox21": load_Tox21_dataset,
        "SIDER": load_SIDER_dataset,
        "ESOL": load_ESOL_dataset,
        "FreeSolv": load_FreeSolv_dataset,
        "Lipophilicity": load_Lipophilicity_dataset,
    }

    if dataset not in loader_map:
        raise ValueError(
            f"Unknown dataset: {dataset}. Available datasets are: {list(loader_map.keys())}"
        )

    smiles_list, rdkit_mol_objs, labels = loader_map[dataset](dataset)

    mol_graph = {}
    drug_id = []

    for idx, mol in enumerate(rdkit_mol_objs):
        key = str(idx)
        mol_graph[key] = mol_to_graphs(mol)
        drug_id.append(key)

    drug_id = np.asarray(drug_id)
    prop = np.asarray(labels, dtype=np.float32)

    dataset_obj = MolDataset(
        root="data",
        dataset=dataset,
        drug_key=drug_id,
        y=prop,
        mol_graph=mol_graph,
    )

    return dataset_obj, smiles_list