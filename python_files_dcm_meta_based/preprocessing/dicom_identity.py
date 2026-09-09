from __future__ import annotations

import re
from typing import Optional, Sequence


def UID_generator(pydicom_obj):
    UID_def = f"{str(pydicom_obj[0x0010,0x0010].value)} ({str(pydicom_obj[0x0010,0x0020].value)})"
    return UID_def


def extract_fraction_number(value: str, allowed_prefixes: Sequence[str]) -> Optional[int]:
    """Return the first case-insensitive integer following an allowed prefix."""
    if not value or not allowed_prefixes:
        return None
    prefix_pattern = "|".join(re.escape(str(prefix)) for prefix in allowed_prefixes)
    match = re.search(r"(?:{})\s*(\d+)".format(prefix_pattern), str(value), re.IGNORECASE)
    return int(match.group(1)) if match else None
