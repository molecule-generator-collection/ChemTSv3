import copy
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem import inchi
from rdkit.Chem import Draw, rdDepictor, rdFingerprintGenerator
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from IPython.display import display

def mol_validity_check(mol: Mol):
    if mol is None or mol.GetNumAtoms() == 0:
        return False
    _mol = copy.deepcopy(mol) # _mol = Chem.Mol(mol)
    if Chem.SanitizeMol(_mol, catchErrors=True).name != "SANITIZE_NONE":
        return False
    return True

def convert_to_canonical(smiles: str, main_mol_only: bool=False) -> str | None:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        if main_mol_only:
            mol = get_main_mol(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None

def is_same_mol(mol1: Mol, mol2: Mol, options=None):
    inchi1 = inchi.MolToInchiKey(mol1, options)
    inchi2 = inchi.MolToInchiKey(mol2, options)
    return inchi1 == inchi2

def get_main_mol(mol):
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    main_mol = max(frags, key=lambda m: m.GetNumAtoms())
    return main_mol

def remove_isotopes(mol):
    for atom in mol.GetAtoms():
        atom.SetIsotope(0)
    return mol

def print_atoms_and_labels(mol: Mol):
    for a in mol.GetAtoms():
        text = a.GetSymbol() + ", MapNum: " + str(a.GetAtomMapNum())
        if a.HasProp('atomLabel'):
            text += ", label: " + a.GetProp("atomLabel")
        print(text)

def draw_mol(mol: Mol, width=300, height=300, all_prop=False):
    if all_prop:
        for a in mol.GetAtoms():
            if a.HasProp("atomLabel"):
                label = a.GetProp("atomLabel")
                label += "_" + a.GetProp("polymerName")
                label += "_" + a.GetProp("monomerIndex")
                a.SetProp("atomLabel", label)

    rdDepictor.SetPreferCoordGen(True)
    rdDepictor.Compute2DCoords(mol, clearConfs=True)
    display(Draw.MolToImage(mol, size = (width, height)))
    
def top_k_df(df: str | pd.DataFrame, k: int, target: str="reward") -> list[str]:
    if isinstance(df, (str, os.PathLike)):
        df = pd.read_csv(df)
    elif isinstance(df, pd.DataFrame):
        df = df.copy()
    else:
        raise TypeError("Input DataFrame or path")

    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=[target, "key"])

    sort_cols = [target]
    ascending = [False]
    for col in ("time", "order"):
        if col in df.columns:
            sort_cols.append(col)
            ascending.append(True)

    df = df.sort_values(sort_cols, ascending=ascending, kind="stable")

    return df.head(k)

def draw_mols(df: pd.DataFrame, legends: list[str], mols_per_row=5, size=(200, 200), max_mols=50, str2mol_func=None):
    mols = []
    legend_strings = []
    for _, row in df.iterrows():
        if str2mol_func is None:
            mol = Chem.MolFromSmiles(row["key"])
        else:
            mol = str2mol_func(row["key"])
        legend = ""
        for val in legends:
            if isinstance(row[val], float):
                legend += f"{val}: {row[val]:.4f}\n"
            else:
                legend += f"{val}: {row[val]:}\n"
        mols.append(mol)
        legend_strings.append(legend)
    display(Draw.MolsToGridImage(mols, molsPerRow=mols_per_row, subImgSize=size, legends=legend_strings, maxMols=max_mols, useSVG=True))
    
def append_similarity_to_df(df: pd.DataFrame, goal_smiles: str, radius=2, fp_size=2048, name: str="similarity"):
    mfgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)

    goal = Chem.MolFromSmiles(goal_smiles)
    goal_fp = mfgen.GetFingerprint(goal)

    def calc_similarity(smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = mfgen.GetFingerprint(mol)
        return TanimotoSimilarity(fp, goal_fp)

    df[name] = df["key"].apply(calc_similarity)
    return df