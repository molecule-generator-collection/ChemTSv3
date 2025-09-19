from rdkit.Chem import Mol, FilterCatalog
from rdkit.Chem.FilterCatalog import FilterCatalogParams
from filter import MolFilter

class CatalogFilter(MolFilter):
    def __init__(self, catalogs: list[str]=["A", "B", "C"]):
        self.catalogs = [f.upper() for f in catalogs]
        
        params = FilterCatalogParams()
        for f in self.catalogs:
            if f == "PAINS_A":
                params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
            elif f == "PAINS_B":
                params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
            elif f == "PAINS_C":
                params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
            elif f == "PAINS":
                params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
            elif f == "BRENK":
                params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
            elif f == "NIH":
                params.AddCatalog(FilterCatalogParams.FilterCatalogs.NIH)
            elif f == "ZINC":
                params.AddCatalog(FilterCatalogParams.FilterCatalogs.ZINC)
            elif f == "ALL":
                params.AddCatalog(FilterCatalogParams.FilterCatalogs.ALL)
            else:
                raise ValueError(f"Invalid catalog name: {f}.")
        self.filter_catalogs = FilterCatalog.FilterCatalog(params)

    # implement
    def mol_check(self, mol: Mol) -> bool:
        return not self.filter_catalogs.HasMatch(mol)