"""Reference-set agreement with an explicit identity definition."""


def compare_identifier_sets(reference, observed, *, identity_definition):
    if not isinstance(identity_definition, str) or not identity_definition.strip():
        raise ValueError("Set agreement requires an explicit identity definition")
    values = []
    for identifiers in [reference, observed]:
        identifiers = list(identifiers)
        if any(not isinstance(value, str) or not value.strip() for value in identifiers):
            raise ValueError("Comparison identifiers must be nonempty strings")
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Duplicate identifiers: define and deduplicate the comparison population explicitly")
        values.append(set(identifiers))
    expected, actual = values
    union = expected | actual
    return {
        "identity_definition": identity_definition,
        "reference_count": len(expected),
        "observed_count": len(actual),
        "intersection_count": len(expected & actual),
        "union_count": len(union),
        "jaccard": len(expected & actual) / len(union) if union else None,
        "reference_only": sorted(expected - actual),
        "observed_only": sorted(actual - expected),
    }
