"""Compare documented scaffold operations across RDKit versions, without changing data.

The fragment-parent calculation is a labelled counterfactual diagnostic, not the
article protocol or a proposed default. No records are removed to match 204.
"""
from pathlib import Path
import argparse
import collections
import csv
import hashlib
import json
import platform
from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold


def core(mol, *, isomeric=True):
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    Chem.RemoveStereochemistry(scaffold)
    return Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=isomeric)


def diagnose(source):
    rows = list(csv.DictReader(source.open(newline='')))
    if len(rows) != 4015:
        raise ValueError('Expected the complete frozen 4015-row table')
    summaries = []
    details = {}
    for role in ['organocatalyst', 'electrophile', 'nucleophile']:
        identities = {}
        for index, row in enumerate(rows):
            mol = Chem.MolFromSmiles(row[role])
            if mol is None:
                raise ValueError(f'Invalid {role} in source row {index + 1}')
            canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
            axial = row['axial_configuration'] if role == 'organocatalyst' else None
            key = (canonical, axial)
            if key not in identities:
                fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                ordered = sorted(fragments, key=lambda fragment: (
                    -fragment.GetNumHeavyAtoms(), Chem.MolToSmiles(fragment)))
                parent = ordered[0]
                identities[key] = {
                    'source_smiles': row[role], 'canonical_smiles': canonical,
                    'axial_configuration': axial, 'source_rows': [],
                    'paper_documented_raw_scaffold': core(mol),
                    'mcp_canonical_scaffold': core(Chem.MolFromSmiles(canonical), isomeric=False),
                    'largest_fragment_counterfactual_scaffold': core(parent),
                    'fragment_count': len(fragments),
                    'fragment_heavy_atom_counts': [f.GetNumHeavyAtoms() for f in ordered],
                    'fragment_parent_size_tie': len(ordered)>1 and ordered[0].GetNumHeavyAtoms()==ordered[1].GetNumHeavyAtoms(),
                    'fragment_scaffolds': [core(f) for f in ordered],
                }
            identities[key]['source_rows'].append(index + 1)
        records = sorted(identities.values(), key=lambda r:(r['canonical_smiles'],r['axial_configuration'] or ''))
        methods = {}
        for method in ['paper_documented_raw_scaffold', 'mcp_canonical_scaffold', 'largest_fragment_counterfactual_scaffold']:
            counts = collections.Counter(r[method] for r in records)
            methods[method] = {'nonempty_scaffolds':sum(bool(k) for k in counts),
                               'acyclic_identities':counts.get('',0)}
        summaries.append({'role':role,'source_records':len(rows),'unique_identities':len(records),
                          'methods':methods,
                          'raw_vs_mcp_different_identities':sum(r['paper_documented_raw_scaffold']!=r['mcp_canonical_scaffold'] for r in records),
                          'fragment_counterfactual_changed_identities':sum(r['paper_documented_raw_scaffold']!=r['largest_fragment_counterfactual_scaffold'] for r in records)})
        details[role] = records
    return {'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'python':platform.python_version(),'platform':platform.platform(),'rdkit':rdBase.rdkitVersion,
            'source_records':len(rows),'summaries':summaries,'identities':details,
            'scope':'Diagnostic comparison only; no production rule, cohort, or published value changed',
            'reference_versions':{'paper_scaffold_methods':'2025.09.2','repository_environment_yml':'2025.9.6','mcp':'2026.3.3'},
            'counterfactual_warning':'Largest fragment is not documented in the paper scaffold method. Ties are reported; this is not an identity-resolution rule.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]/'chem_repro/data/mannich/mannich_dataset.csv'
    result = diagnose(source)
    if result['source_sha256']!='903dd5d69bfe6c3b6a93bbde344f4df3bd7186a9f775eae18c954ffb252fd4a7':
        raise ValueError('Source table differs from the declared reference')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['source_sha256','python','rdkit','summaries','scope']}))
