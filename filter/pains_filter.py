from rdkit.Chem import Mol, FilterCatalog
from rdkit.Chem.FilterCatalog import FilterCatalogParams
from filter import MolFilter

class PainsFilter(MolFilter):
    """
    ref: Baell et al. "New Substructure Filters for Removal of Pan Assay Interference Compounds (PAINS) from Screening Libraries and for Their Exclusion in Bioassays", Medical Chemistry (2009)
    """
    def __init__(self, families: list[str]=["A", "B", "C"]):
        self.families = [f.upper() for f in families]

    # implement
    def mol_check(self, mol: Mol) -> bool:
        params = FilterCatalogParams()
        for f in self.families:
            if f == "A":
                params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
            elif f == "B":
                params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
            elif f == "C":
                params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
            else:
                raise ValueError("family must be either 'A', 'B', or 'C'.")
        filter_catalogs = FilterCatalog.FilterCatalog(params)
        return not filter_catalogs.HasMatch(mol)