import numpy as np
from rdkit import Chem
from rdkit.Chem import rdChemReactions

from chem_repro.common.reactions import structural_reaction_fingerprint


def test_structural_params_match_function_not_empty_constructor_defaults():
    reaction = rdChemReactions.ChemicalReaction()
    reaction.AddReactantTemplate(Chem.MolFromSmiles("CC=O"))
    reaction.AddProductTemplate(Chem.MolFromSmiles("CCO"))
    reaction.AddAgentTemplate(Chem.MolFromSmiles("N1CCCC1"))
    expected = rdChemReactions.CreateStructuralFingerprintForReaction(reaction)
    actual = structural_reaction_fingerprint(["CC=O"], ["CCO"], ["N1CCCC1"])
    assert actual.GetNumBits() == 4096 and actual.ToBitString() == expected.ToBitString()
    wrong = rdChemReactions.CreateStructuralFingerprintForReaction(
        reaction, rdChemReactions.ReactionFingerprintParams()
    )
    assert wrong.GetNumBits() == 2048


def test_agent_inclusion_and_direction_are_explicit():
    a = np.asarray(structural_reaction_fingerprint(["CC=O"], ["CCO"], ["N1CCCC1"]))
    b = np.asarray(structural_reaction_fingerprint(["CC=O"], ["CCO"], ["c1ccccc1"]))
    assert not np.array_equal(a, b)
    a = structural_reaction_fingerprint(["CC=O"], ["CCO"], ["N1CCCC1"], include_agents=False)
    b = structural_reaction_fingerprint(["CC=O"], ["CCO"], ["c1ccccc1"], include_agents=False)
    reverse = structural_reaction_fingerprint(["CCO"], ["CC=O"], include_agents=False)
    assert a.ToBitString() == b.ToBitString()
    assert a.ToBitString() != reverse.ToBitString()
