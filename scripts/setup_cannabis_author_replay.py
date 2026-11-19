"""Assemble a portable, unchanged-author-code training project without fitting anything.

python scripts/setup_cannabis_author_replay.py --project-dir /absolute/new/project \
    --output-dir /absolute/new/results
Run the printed command through the configured compute runner after committing.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

ROOT=Path(__file__).resolve().parents[1]
VENDOR=ROOT/'chem_repro/vendor/canpest_author'
DATA=ROOT/'chem_repro/data/cannabis'


def assemble(project,output):
    project=Path(project).resolve();output=Path(output).resolve()
    if project.exists() and any(project.iterdir()):
        raise ValueError('Use an empty project directory to preserve previous source and results')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use an empty result directory for new author-code fits')
    source=json.loads((VENDOR/'source_manifest.json').read_text())
    project.mkdir(parents=True,exist_ok=True)
    for name,digest in source['files'].items():
        src=VENDOR/name
        if hashlib.sha256(src.read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Original author code checksum differs: {name}')
        dest=project/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src,dest)
    deposited=json.loads((DATA/'manifest.json').read_text())
    for name,spec in deposited['files'].items():
        src=DATA/name
        if hashlib.sha256(src.read_bytes()).hexdigest()!=spec['sha256']:
            raise ValueError(f'Deposited input checksum differs: {name}')
        if name in {'split_registry.csv','metadata.json'}:dest=project/'data/splits'/name
        elif name.endswith('_residue_matrix.csv'):dest=project/'source_inputs/residue_matrices'/name
        else:dest=project/'data'/name
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
    original={'repository':'chemagents/canpest','commit':source['revision'],
        'files':{str(p.relative_to(project)):hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(project.rglob('*')) if p.is_file()}}
    (project/'author_source_manifest.json').write_text(json.dumps(original,indent=2))
    shutil.copyfile(VENDOR/'replay.py',project/'replay.py')
    config=json.loads((VENDOR/'replay_defaults.json').read_text());config['output_dir']=str(output)
    (project/'replay_config.json').write_text(json.dumps(config,indent=2))
    (project/'requirements.txt').write_text('numpy==2.2.6\npandas==2.3.3\nscipy==1.15.3\nscikit-learn==1.9.0\nrdkit==2026.3.3\ntorch==2.10.0\ncatboost==1.2.8\njoblib==1.6.0\n')
    (project/'.gitignore').write_text('__pycache__/\n*.pyc\n.venv/\ncatboost_info/\n')
    print(json.dumps({'project':str(project),'output':str(output),'command':'python -u replay.py',
        'source_commit':source['revision'],'n_splits':len(config['split_ids']),
        'mode':'new fits of original deposited RMT95 code; not final RTE390 mixture'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-dir',required=True)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args();assemble(args.project_dir,args.output_dir)
