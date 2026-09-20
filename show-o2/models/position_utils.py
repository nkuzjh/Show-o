"""Small spatial-position helpers shared by Show-o2 resolution paths."""

from __future__ import annotations


def position_table_matches_grid(position_embedding, patch_height: int, patch_width: int) -> bool:
    """Return whether the loaded position table directly matches a patch grid."""
    return position_embedding.weight.shape[0] == int(patch_height) * int(patch_width)
