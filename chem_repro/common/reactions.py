"""Role-preserving structural reaction fingerprints; no OCR or inferred atom mapping."""

from rdkit import Chem
from rdkit.Chem import rdChemReactions


def structural_reaction_fingerprint(reactants, products, agents=(), *, include_agents=True):
    if not reactants or not products:
        raise ValueError("Both reactant and product roles must be present")
    reaction = rdChemReactions.ChemicalReaction()
    for role, smiles in [("reactant", reactants), ("product", products), ("agent", agents)]:
        for value in smiles:
            mol = Chem.MolFromSmiles(value)
            if mol is None:
                raise ValueError(f"Invalid {role} structure")
            {
                "reactant": reaction.AddReactantTemplate,
                "product": reaction.AddProductTemplate,
                "agent": reaction.AddAgentTemplate,
            }[role](mol)
    # These are the structural-function defaults in RDKit Release_2026_03_3,
    # NOT the different defaults of the empty ReactionFingerprintParams constructor.
    params = rdChemReactions.ReactionFingerprintParams()
    params.fpType = rdChemReactions.FingerprintType.PatternFP
    params.fpSize = 4096
    params.includeAgents = include_agents
    params.bitRatioAgents = 0.2
    params.nonAgentWeight = params.agentWeight = 1
    return rdChemReactions.CreateStructuralFingerprintForReaction(reaction, params)
