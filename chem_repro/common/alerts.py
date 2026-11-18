"""RDKit structural-alert catalog checks, separate from experimental toxicity."""

from functools import lru_cache

from rdkit import Chem
from rdkit.Chem import FilterCatalog

from .chemistry import canonical_smiles


@lru_cache(maxsize=2)
def _catalog(name):
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(getattr(FilterCatalog.FilterCatalogParams.FilterCatalogs, name))
    return FilterCatalog.FilterCatalog(params)


@lru_cache(maxsize=16384)
def structural_alerts(smiles):
    mol = Chem.MolFromSmiles(canonical_smiles(smiles))
    output = {}
    for name in ["PAINS", "BRENK"]:
        hits = sorted(entry.GetDescription() for entry in _catalog(name).GetMatches(mol))
        output[name.lower() + "_count"] = len(hits)
        output[name.lower() + "_names"] = hits
    return output
