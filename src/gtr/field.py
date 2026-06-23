"""Decision-field vocabulary for ablations."""

from __future__ import annotations


FIELD_VARIANTS = {
    "additive": "raw_plus_field_no_residual",
    "interaction": "full_hybrid",
    "residual": "raw_head_on_z_prime",
}
