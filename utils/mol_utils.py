from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem import inchi
from rdkit.Chem import Draw, rdDepictor
from IPython.display import display

def is_same_mol(mol1: Mol, mol2: Mol, options=None):
    inchi1 = inchi.MolToInchiKey(mol1, options)
    inchi2 = inchi.MolToInchiKey(mol2, options)
    return inchi1 == inchi2

def standardize_mol(mol):
    #mol = Chem.RemoveHs(mol)
    mol = get_main_mol(mol)
    mol = remove_isotopes(mol)
    return mol

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