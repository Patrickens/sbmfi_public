import pandas as pd

_ACN = "C2H3N"
_IsoProp = "C3H8O"
_Methanol = "CH3OH"
_DMSO = "C2H6OS"

_Hac = "C2H4O2"
_TFA = "CF3CO2H"
_FA = "H2CO2"

_default_adducts = [
    ("M+", 1, "", "", 1),
    ("M+H", 1, "H", "", 1),
    ("M+NH4", 1, "NH4", "", 1),
    ("M+H+NH4", 1, "HNH4", "", 2),
    ("M+Na", 1, "Na", "", 1),
    ("M+H-2H2O", 1, "H", "(H2O)2", 1),
    ("M+H-H2O", 1, "H", "H2O", 1),
    ("M+K", 1, "K", "", 1),
    ("M+ACN+H", 1, "C2H3NH", "", 1),
    ("M+2ACN+H", 1, f"({_ACN})2H", "", 1),
    ("M+2ACN+2H", 1, f"({_ACN})2H2", "", 2),
    ("M+3ACN+2H", 1, f"({_ACN})3H2", "", 2),
    ("M+ACN+2H", 1, f"({_ACN})1H2", "", 2),
    ("M+ACN+Na", 1, f"({_ACN})1Na", "", 1),
    ("M+2Na-H", 1, "Na2", "H", 1),
    ("M+Li", 1, "Li", "", 1),
    ("M+CH3OH+H", 1, f"{_Methanol}H", "", 1),
    ("M+2H", 1, "H2", "", 2),
    ("M+H+Na", 1, "HNa", "", 2),
    ("M+H+K", 1, "HK", "", 2),
    ("M+3H", 1, "H3", "", 3),
    ("M+2H+Na", 1, "(H2)1Na", "", 3),
    ("M+2Na", 1, "Na2", "", 2),
    ("M+2K-H", 1, "K2", "H", 1),
    ("M+3Na", 1, "Na3", "", 3),
    ("M+2Na+H", 1, "(Na2)1H", "", 3),
    ("M+IsoProp+H", 1, f"({_IsoProp})1H", "", 1),
    ("M+IsoProp+Na+H", 1, f"({_IsoProp})1NaH", "", 1),
    ("M+DMSO+H", 1, f"({_DMSO})1H", "", 1),
    ("M-", 1, "", "", -1),
    ("M-H", 1, "", "H", -1),
    ("M-2H", 1, "", "H2", -2),
    ("M-3H", 1, "", "H3", -3),
    ("M-H2O-H", 1, "", "H2OH", -1),
    ("M+Na-2H", 1, "Na", "H2", -1),
    ("M+Cl", 1, "Cl", "", -1),
    ("M+K-2H", 1, "K", "H2", -1),
    ("M+KCl-H", 1, "KCl", "H", -1),
    ("M+FA-H", 1, _FA, "H", -1),
    ("M+F", 1, "F", "", -1),
    ("M+Hac-H", 1, _Hac, "H", -1),
    ("M+Br", 1, "Br", "", -1),
    ("M+TFA-H", 1, _TFA, "H", -1),
    ("2M+H", 2, "H", "", 1),
    ("2M+NH4", 2, "NH4", "", 1),
    ("2M+Na", 2, "Na", "", 1),
    ("2M+K", 2, "K", "", 1),
    ("2M+ACN+H", 2, f"({_ACN})1H", "", 1),
    ("2M+ACN+Na", 2, f"({_ACN})1Na", "", 1),
    ("2M-H", 2, "", "H", -1),
    ("2M+FA-H", 2, _FA, "H", -1),
    ("2M+Hac-H", 2, _Hac, "H", -1),
    ("3M-H", 3, "", "H", -1),
    ("M", 1, "", "", 0),
]
emzed_adducts = pd.DataFrame(
    _default_adducts, columns=["adduct_name", "m_multiplier", "adduct_add", "adduct_sub", "z"]
).set_index(keys='adduct_name')