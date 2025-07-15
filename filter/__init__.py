from .base import Filter, MolFilter, ValueFilter, MolValueFilter
from .aromatic_ring_filter import AromaticRingFilter
from .attachment_points_filter import AttachmentPointsFilter
from .charge_filter import ChargeFilter
from .connectivity_filter import ConnectivityFilter
from .hba_filter import HBAFilter
from .hbd_filter import HBDFilter
from .log_p_filter import LogPFilter
from .num_atoms_filter import NumAtomsFilter
from .pains_filter import PainsFilter
from .radical_filter import RadicalFilter
from .ring_bond_filter import RingBondFilter
from .ring_size_filter import MaxRingSizeFilter, MinRingSizeFilter
from .rotatable_bonds_filter import RotatableBondsFilter
from .tpsa_filter import TPSAFilter
from .validity_filter import ValidityFilter
from .weight_filter import WeightFilter

# lazy import
def __getattr__(name):
    if name == "PubChemFilter":
        from .pubchem_filter import PubChemFilter
        return PubChemFilter
    if name == "LipinskiFilter":
        from .lipinski_filter import LipinskiFilter
        return LipinskiFilter
    if name == "SAScoreFilter":
        from .sa_score_filter import SAScoreFilter
        return SAScoreFilter
    raise AttributeError(f"module {__name__} has no attribute {name}")