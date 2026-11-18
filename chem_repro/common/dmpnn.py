"""Chemprop components for an explicit reconstruction of Cannabis Table S1.

The installed Chemprop version, raw descriptor panel, train-only normalization,
norm constant and total-vs-hidden FFN layer convention remain explicit. These
choices do not recover missing author checkpoints or final cohort membership.
"""

import numpy as np
import torch
from chemprop import data, models, nn
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold


class MeanSumNormAggregation(nn.agg.Aggregation):
    """Table S1's mean, sum, norm, in that order; norm uses an explicit constant."""

    def __init__(self, dim=0, norm=100.0):
        super().__init__(dim=dim)
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("Aggregation normalization must be finite and positive")
        self.norm = float(norm)
        self.hparams["norm"] = self.norm
        self.mean = nn.MeanAggregation(dim=dim)
        self.sum = nn.SumAggregation(dim=dim)

    def forward(self, H, batch):
        summed = self.sum(H, batch)
        return torch.cat([self.mean(H, batch), summed, summed / self.norm], dim=1)


class TrainOnlyDescriptorScaler:
    """Source-style median imputation, standardization and clipping fitted on train only."""

    def fit(self, x):
        x = np.asarray(x, dtype=float).copy()
        x[~np.isfinite(x)] = np.nan
        self.median = np.asarray([np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else 0.0
                                  for col in x.T])
        clean = np.where(np.isnan(x), self.median, x)
        self.mean, self.std = clean.mean(0), clean.std(0)
        self.keep = self.std > 1e-6
        self.std[~self.keep] = 1.0
        if not self.keep.any():
            raise ValueError("No variable molecular descriptors in the training partition")
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=float)
        clean = np.where(np.isfinite(x), x, self.median)
        return np.clip((clean - self.mean) / self.std, -5.0, 5.0)[:, self.keep].astype(np.float32)

    def to_dict(self):
        return {name: getattr(self, name).tolist() for name in ["median", "mean", "std", "keep"]}

    @classmethod
    def from_dict(cls, values):
        out = cls()
        for name in ["median", "mean", "std", "keep"]:
            setattr(out, name, np.asarray(values[name], dtype=bool if name == "keep" else float))
        return out


def source_structure_view(smiles):
    """Keep source connectivity, charges/isotopes/fragments; remove only declared stereo labels."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("Unparseable/empty source molecule; no silent row exclusion")
    canonical_source = Chem.MolToSmiles(mol, isomericSmiles=True)
    fragment_count = len(Chem.GetMolFrags(mol))
    charge = Chem.GetFormalCharge(mol)
    Chem.RemoveStereochemistry(mol)
    # isomericSmiles=True retains isotopes after explicitly removing stereo.
    view = Chem.MolToSmiles(mol, isomericSmiles=True)
    core = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return mol, {"canonical_source": canonical_source, "analysis_smiles": view, "scaffold": core,
                 "stereo_removed": canonical_source != view, "source_fragments": fragment_count,
                 "source_formal_charge": charge}


def descriptor_matrix(molecules):
    panel = list(Descriptors._descList)
    values = np.full((len(molecules), len(panel)), np.nan)
    for i, mol in enumerate(molecules):
        for j, (_, fn) in enumerate(panel):
            try:
                values[i, j] = fn(mol)
            except (ValueError, RuntimeError, OverflowError, ZeroDivisionError):
                pass
    values[~np.isfinite(values)] = np.nan
    return values, [name for name, _ in panel]


def create_model(n_descriptors, *, hidden=300, depth=3, dropout=0.1,
                 ffn_hidden=300, ffn_linear_layers=2, norm=100.0):
    if ffn_linear_layers < 2:
        raise ValueError("Use at least one hidden and one output linear layer")
    mp = nn.BondMessagePassing(d_h=hidden, depth=depth, dropout=dropout, undirected=False)
    agg = MeanSumNormAggregation(norm=norm)
    ffn = nn.BinaryClassificationFFN(input_dim=3 * hidden + n_descriptors,
                                     hidden_dim=ffn_hidden, n_layers=ffn_linear_layers - 1,
                                     dropout=dropout)
    return models.MPNN(mp, agg, ffn, batch_norm=False, metrics=[nn.BinaryAUROC()],
                       warmup_epochs=2, init_lr=1e-4, max_lr=1e-3, final_lr=1e-4)


def molecule_dataset(molecules, labels, descriptors):
    points = [data.MoleculeDatapoint(mol=mol, y=np.asarray([y],dtype=np.float32),
                                     x_d=np.asarray(x,dtype=np.float32))
              for mol, y, x in zip(molecules, labels, descriptors, strict=True)]
    dataset = data.MoleculeDataset(points)
    dataset.cache = True
    return dataset


def build_loader(dataset, *, batch_size=50, shuffle=False, seed=42):
    """Retain a last batch of one; Chemprop 2.2.1's convenience loader drops it.

    Batch normalization is disabled in this protocol, so singleton batches are
    valid. Dropping a test row would break complete OOF/query coverage.
    """
    from chemprop.data.collate import collate_batch
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                       generator=torch.Generator().manual_seed(seed),
                                       num_workers=0, collate_fn=collate_batch, drop_last=False)
