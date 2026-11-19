import numpy as np
from chem_repro.mannich import space


def test_all_identity_rows_retained_across_chirality_modes(monkeypatch):
    monkeypatch.setattr(space, "fingerprint_tsne", lambda rows, **kwargs: (np.zeros((len(rows), 2)), 0.0, len(rows)))
    space._chemical_space.cache_clear()
    try:
        a = space.chemical_space("catalysts", "notebook")
        b = space.chemical_space("catalysts", "chirality_sensitivity")
        ca, cb = [r.answer.tables["coordinates"] for r in (a,b)]
        assert len(ca.rows) == len(cb.rows) == 468
        audit_a, audit_b = [r.answer.tables["identity_mapping_audit"].rows[0] for r in (a,b)]
        assert audit_a["identity_mapping_sha256"] == audit_b["identity_mapping_sha256"]
        assert audit_a["unique_projection_ids"] == 468
        assert ca.rows == cb.rows
        assert a.metadata.protocol_parameters["include_chirality"] is False
        assert b.metadata.protocol_parameters["include_chirality"] is True
        evidence = a.answer.tables["catalyst_method_source_evidence"]
        assert evidence.evidence_kind == "source_record"
        assert evidence.rows[0]["paper_quote"] == "chirality-sensitive Morgan fingerprints"
        assert "includeChirality" not in evidence.rows[0]["generator_code"]
        reagents = space.chemical_space("reagents", "notebook")
        assert len(reagents.answer.tables["coordinates"].rows) == 1106
        assert reagents.answer.tables["identity_mapping_audit"].rows[0]["source_identity_rows"] == 1106
    finally:
        space._chemical_space.cache_clear()
