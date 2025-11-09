from stimojo.xzencoding import XZEncoding


# TODO: check for commutation
fn commutes(p1: XZEncoding, p2: XZEncoding) -> Bool:
    return True


fn commutes(p1: String, p2: String) -> Bool:
    return True


# TODO: partition paulis into commuting groups
fn sort_into_commuting_groups(
    pauli_encodings: List[String],
) -> List[List[String]]:
    ...


fn sort_into_commuting_groups(
    pauli_encodings: List[XZEncoding],
) -> List[List[XZEncoding]]:
    ...
