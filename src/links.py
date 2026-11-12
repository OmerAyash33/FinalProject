"""
Clickable results: open a file / folder with the system's default program, and print the end-of-run links.

- `open_path(path)` is what a click on a link in the app (src/app.py) does.
- `print_result_links(...)` ends a command-line run (src/main.py): the absolute path (PyCharm and most terminals make an existing path
  clickable) and the same path as a `file:///` link (Windows Terminal, VS Code, PyCharm and browsers open that).
  Inside the app (FOOTBALL_IN_APP=1) only the plain path lines are printed: the app turns them into links itself.
"""
import os
import subprocess
import sys
from pathlib import Path


def file_uri(path):
    """`C:\\a b\\c.mp4` -> `file:///C:/a%20b/c.mp4` (what browsers and terminals open on click)."""
    return Path(os.path.abspath(path)).as_uri()


def open_path(path):
    """Open a file (video player, spreadsheet, ...) or a folder (file explorer) with the system default. Raises OSError when that fails."""
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if os.name == "nt":
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def print_result_links(output_dir, master_video=None, highlight_videos=()):
    """Print where the results are, as links. `master_video`: the one-reel file (printed only when it exists)."""
    in_app = os.environ.get("FOOTBALL_IN_APP") == "1"
    items = []
    if master_video and os.path.exists(master_video):
        items.append(("Highlight reel (all videos in one)", master_video))
    items += [("Highlights of " + os.path.basename(v)[:-len("_highlights.mp4")], v) for v in highlight_videos if os.path.exists(v)]
    items.append(("Results folder", output_dir))
    print(f"\n{'=' * 60}\nDONE. Your results (click a link to open it):\n{'=' * 60}", flush=True)
    for label, path in items:
        path = os.path.abspath(path)
        print(f"  {label}: {path}", flush=True)
        if not in_app:
            print(f"      {file_uri(path)}", flush=True)
