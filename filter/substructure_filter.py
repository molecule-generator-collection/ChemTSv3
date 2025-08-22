from rdkit import Chem
from rdkit.Chem import Mol
from filter import MolFilter

class SubstructureFilter(MolFilter):
    def __init__(self, smiles: str | list[str]=None, smarts: str | list[str]=None, kekulize: bool=False, preserve=True):
        """
        Args:
            preserve: If True, pass molecules WITH all of the specified substructures. If False, pass molecules WITHOUT any of the specified substructures.
        """
        self.targets = []
        if smiles is None and smarts is None:
            raise ValueError("Specify either 'smiles' or 'smarts'.")
        elif smiles is not None and smarts is not None:
            raise ValueError("Specify one of 'smiles' or 'smarts', not both.")
        elif smiles is not None:
            if isinstance(smiles, str):
                smiles = [smiles]
            for s in smiles:
                mol = Chem.MolFromSmiles(s)
                if mol is None:
                    raise ValueError(f"Invalid SMILES: {s}")
                self.targets.append(mol)
        else:
            if isinstance(smarts, str):
                smarts = [smarts]
            for s in smarts:
                mol = Chem.MolFromSmarts(s)
                if mol is None:
                    raise ValueError(f"Invalid SMILES: {s}")
                self.targets.append(mol)
        
        self.kekulize = kekulize
        self.preserve = preserve
        
    def _prep(self, mol):
        if not self.kekulize:
            return mol
        else:
            m = Chem.Mol(mol)
            try:
                Chem.Kekulize(m, clearAromaticFlags=True)
            except Exception:
                pass
            return m

    # implement
    def mol_check(self, mol: Mol) -> bool:
        m = self._prep(mol)
        if self.preserve:
            return all(m.HasSubstructMatch(target) for target in self.targets)
        else:
            return not any(m.HasSubstructMatch(target) for target in self.targets)