"""
The user's own answer to "which shirt colour is OUR team in this session?", stored in `team_colors.json` (next to the program,
see config.APP_ROOT) and written by src/team_color_picker.py.

Format:  {"<session>": {"ours": [[b, g, r], ...], "theirs": [[b, g, r], ...]}}
Every row is the median chest colour (the very quantity Stage 2 compares) of one player the user clicked: left click = our team,
right click = an opponent. Several clicks are kept, the reference colour is their median (see team_utils.TeamModel.from_reference).
`session` is the footage folder name without any "_v2" suffix (team_utils.session_of).
"""
import json
import os

import numpy as np

from app_paths import APP_ROOT

TEAM_COLORS_PATH = os.path.join(APP_ROOT, "team_colors.json")


def _read(path=None):
    path = path or TEAM_COLORS_PATH
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def load_reference(session, path=None):
    """dict(ours=(n, 3) array, theirs=(m, 3) array) of B,G,R samples, or None when the user has not picked a colour for this session."""
    entry = _read(path).get(session)
    if not entry or not entry.get("ours"):
        return None
    return {"ours": np.asarray(entry["ours"], dtype=float).reshape(-1, 3),
            "theirs": np.asarray(entry.get("theirs") or [], dtype=float).reshape(-1, 3)}


def save_reference(session, ours, theirs, path=None):
    """Store (replace) the samples of one session; the other sessions in the file are kept. Written atomically."""
    path = path or TEAM_COLORS_PATH
    data = _read(path)
    data[session] = {"ours": [[round(float(v), 1) for v in c] for c in ours], "theirs": [[round(float(v), 1) for v in c] for c in theirs]}
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)


def clear_reference(session, path=None):
    path = path or TEAM_COLORS_PATH
    data = _read(path)
    if session in data:
        del data[session]
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)
        os.replace(tmp, path)
