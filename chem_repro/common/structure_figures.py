"""Vector chemical structure grids with explicit row identities and printed dimensions."""

import io
import json
import math
import os
import zipfile
from hashlib import sha256

from rdkit import Chem
from rdkit.Chem import Draw, rdCoordGen
from reportlab import rl_config
from reportlab.graphics import renderPDF
from svglib.svglib import svg2rlg

from .artifacts import save_artifact


def structure_grid(smiles, legends, *, columns=3, cell=(450, 320), width_inches=6.75):
    if not smiles or len(smiles) != len(legends):
        raise ValueError("Require one legend per nonempty structure")
    molecules = [Chem.MolFromSmiles(s) for s in smiles]
    if any(m is None for m in molecules):
        raise ValueError("Invalid structure must not disappear from a figure")
    for molecule in molecules:
        rdCoordGen.AddCoords(molecule)
    options = Draw.rdMolDraw2D.MolDrawOptions()
    options.legendFontSize = 25
    options.minFontSize = 16
    options.maxFontSize = 28
    options.legendFraction = 0.19
    options.padding = 0.08
    # Wedges encode point stereochemistry. Extra per-atom R/S text crowds the bridged catalysts;
    # absolute configurations remain in the canonical SMILES and source audit tables.
    options.addStereoAnnotation = False
    options.useBWAtomPalette()
    svg = Draw.MolsToGridImage(
        molecules, molsPerRow=columns, subImgSize=cell, legends=legends, useSVG=True, drawOptions=options
    )
    # RDKit emits paths, including stereo wedges and lettering, rather than a screenshot.
    drawing = svg2rlg(io.BytesIO(svg.encode()))
    if drawing is None or drawing.width <= 0:
        raise RuntimeError("Could not convert the vector structure grid")
    scale = width_inches * 72 / drawing.width
    drawing.scale(scale, scale)
    drawing.width *= scale
    drawing.height *= scale
    rl_config.invariant = 1
    pdf = renderPDF.drawToString(drawing)
    png = Draw.MolsToGridImage(
        molecules,
        molsPerRow=columns,
        subImgSize=cell,
        legends=legends,
        useSVG=False,
        returnPNG=True,
        drawOptions=options,
    )
    return {"svg": svg.encode(), "pdf": pdf, "png": png}


def structure_pages(records, *, smiles_column, legend_fn, name, port, page_size=18, columns=3):
    """ZIP every vector page and metadata; return a first-page preview plus the full bundle."""
    if not records:
        raise ValueError("No structure records to draw")
    manifest = {
        "name": name,
        "n_structures": len(records),
        "page_size": page_size,
        "pdf_width_inches": 6.75,
        "pages": [],
        "rows": records,
    }
    payload = io.BytesIO()
    first_page = None
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as archive:
        for page in range(math.ceil(len(records) / page_size)):
            batch = records[page * page_size : (page + 1) * page_size]
            rendered = structure_grid(
                [r[smiles_column] for r in batch],
                [legend_fn(r) for r in batch],
                columns=columns,
                cell=(450, 390) if page_size == 12 else (450, 320),
            )
            page_record = {
                "page": page + 1,
                "first_record": page * page_size + 1,
                "last_record": page * page_size + len(batch),
                "files": [],
            }
            for ext, data in rendered.items():
                filename = f"{name}_page{page + 1:02d}.{ext}"
                info = zipfile.ZipInfo(filename, date_time=(2026, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, data)
                page_record["files"].append(
                    {"name": filename, "sha256": sha256(data).hexdigest(), "bytes": len(data)}
                )
            manifest["pages"].append(page_record)
            if first_page is None:
                first_page = rendered
        info = zipfile.ZipInfo("manifest.json", date_time=(2026, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info, json.dumps(manifest, indent=2, allow_nan=False))
    base = os.getenv("CHEM_REPRO_PUBLIC_BASE_URL", f"http://127.0.0.1:{port}")
    artifacts = [
        save_artifact(
            payload.getvalue(),
            suffix="zip",
            media_type="application/zip",
            public_base_url=base,
            display_name=name + "_all_pages.zip",
        )
    ]
    for ext, mime in [("svg", "image/svg+xml"), ("pdf", "application/pdf"), ("png", "image/png")]:
        artifacts.append(
            save_artifact(
                first_page[ext],
                suffix=ext,
                media_type=mime,
                public_base_url=base,
                display_name=name + "_page01." + ext,
            )
        )
    return artifacts
