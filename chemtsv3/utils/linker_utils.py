import re
from rdkit import Chem
from rdkit.Chem import AllChem

def calc_morgan_count(mol, r=2, dimension=500):
    info = {}
    _fp = AllChem.GetMorganFingerprint(mol, r, bitInfo=info)
    count_list = [0] * dimension
    for key in info:
        pos = key % dimension
        count_list[pos] += len(info[key])
    return count_list

def add_atom_index_in_wildcard(smiles: str):
    c = iter(range(1, smiles.count('*')+1))
    labeled_smiles = re.sub(r'\*', lambda _: f'[*:{next(c)}]', smiles)
    return labeled_smiles

def link_linker(cores, linker, linker_type="mol", output_type="mol"):
    if linker_type == "mol":
        smi = Chem.MolToSmiles(linker)
    elif linker_type == "smiles":
        smi = linker
    mol_ = Chem.MolFromSmiles(add_atom_index_in_wildcard(smi))
    rwmol = Chem.RWMol(mol_)
    if type(cores) is list:
        cores_mol = [Chem.MolFromSmiles(s) for s in cores]
    else:
        cores_mol = [Chem.MolFromSmiles(s) for s in [cores['ligand_1'], cores['ligand_2']]]
    for m in cores_mol:
        rwmol.InsertMol(m)
    prod = Chem.MolToSmiles(rwmol)
    prod = Chem.molzip(rwmol)
    if output_type == "smiles":
        return Chem.MolToSmiles(prod)
    return prod
