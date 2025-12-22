import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Mol
from chemtsv3.filter import MolFilter

class ROCFilter(MolFilter):
    def __init__(self, filtering_level: int=1, kekulize: bool=False):
        """
        Args:
            filtering_level (int): 
                Specifies the strictness of the filtering rules in three levels (1-3).
                A higher level applies stricter constraints and excludes more structures.
                
                1 (Low): Excludes only structures that should clearly be removed (e.g., chemically unstable or unrealistic structures).
                2 (Medium): Excludes structures that are preferably avoided during lead optimization, but may still be allowed during early-stage molecular generation.
                3 (High): Retains only structures that are generally acceptable and suitable for virtual screening.
        """
    
        if filtering_level not in [1, 2, 3]:
            raise ValueError("filtering_level must be 1, 2, or 3.")
    
        self.filtering_level = filtering_level
        self.kekulize = kekulize
        
        self.targets = []
        
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
        df = pd.read_csv(os.path.join(data_dir, "filtering_substruct_oota_cho.csv"))

        df_filtered = df[df["label"] <= filtering_level]

        for smarts in df_filtered["remove_ss"]:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is None:
                raise ValueError(f"Invalid SMARTS: {smarts}")
            self.targets.append(pattern)
        
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
        return not any(m.HasSubstructMatch(target) for target in self.targets)


        
