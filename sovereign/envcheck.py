"""
Environment / reproducibility checks (pure logic).

The heavy lifting of check_env.py — parsing requirements.txt and comparing
installed versions against the declared minima — lives here so it can be unit
tested without importing torch or transformers.
"""

import re

_REQ_LINE = re.compile(r"^([A-Za-z0-9_.\-]+)\s*(>=|==|>)?\s*([0-9][0-9A-Za-z.\-]*)?")


def parse_requirements(text):
    """Parse a requirements.txt body into {package: min_version_or_None}.

    Only '>=' and '==' constraints yield a minimum; bare packages map to None.
    Comments and blank lines are ignored. Package names are lowercased and
    extras/markers stripped.
    """
    reqs = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        m = _REQ_LINE.match(line)
        if not m:
            continue
        name = m.group(1).lower().split("[")[0]
        op, ver = m.group(2), m.group(3)
        reqs[name] = ver if (op in (">=", "==") and ver) else None
    return reqs


def version_tuple(ver):
    """Turn '2.11.0' / '4.40' into a comparable tuple of ints, ignoring
    non-numeric suffixes like 'rc1' or 'post0'."""
    parts = []
    for chunk in str(ver).split("."):
        num = re.match(r"\d+", chunk)
        parts.append(int(num.group()) if num else 0)
    return tuple(parts)


def meets_minimum(installed, minimum):
    """True if installed >= minimum (both dotted version strings). If minimum
    is None (no constraint) any installed version passes."""
    if minimum is None:
        return True
    if installed is None:
        return False
    a = version_tuple(installed)
    b = version_tuple(minimum)
    width = max(len(a), len(b))
    a += (0,) * (width - len(a))
    b += (0,) * (width - len(b))
    return a >= b
