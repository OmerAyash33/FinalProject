"""
Build the packaged app: dist/FootballHighlights/FootballHighlights.exe  (src/app.py + everything it needs, one folder).

Usage:  .venv/Scripts/python.exe tools/build_exe.py
Takes 10-20 minutes and the folder is several GB (it carries torch with the CUDA libraries); copy the WHOLE folder to another
computer (an NVIDIA GPU makes it ~6x faster, without one it still runs on the CPU, slowly). Not one-file on purpose: a one-file exe of
this size would unpack gigabytes on every start.

What is inside: the trained detector of config/app_paths DETECTOR_VERSION (models/match_detector_<version>.pt), every package of
requirements.txt that the pipeline imports. What is NOT inside: footage, outputs, team_colors.json (the app creates it next to the
exe the first time a colour is picked) and the OpenVINO CPU path (excluded for size; without a GPU the plain PyTorch CPU path is used).
"""
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from app_paths import DETECTOR_VERSION  # noqa: E402

NAME = "FootballHighlights"
README = """Football highlights
===================
1. Double-click FootballHighlights.exe  (keep this whole folder together; the first start takes ~20 seconds).
2. Step 1 - drag your videos into the list (or press "Add videos").
3. Step 2 - check the session name and the output folder.
4. Step 3 - leave "pick our team's shirt colour (recommended)" ticked.
5. Step 4 - press "Start processing". When the extraction is done a window shows players one at a time: left-click a player of YOUR team
   (right-click = opponent, optional; N = skip to the next player). At most 5 players per colour; after 7 clicks in total
   the window closes by itself and the processing goes on.
6. Step 5 - when the run is finished, click the links to open the highlight reel (All_Matches_Master_Highlights.mp4 = everything in one reel)
   or the results folder.

- Step 4 shows how far the run is: "Extracting: 2 of 5 videos done - about 11 min left" and a bar, refreshed every 5 seconds. It is a rough guide
  (the speed is measured as the run goes), and it stops counting while you choose the shirt colour.
- The extraction is the slow stage: a few minutes per 5-minute video on an NVIDIA GPU. While it runs, a "Live view" window shows what the detector sees
  (players in their team colour, ball, goal); untick "Show a live view" in Step 4 if you do not want it. It costs about 10% speed.
- Every highlighted move ends on a summary card; its COACH text is a few sentences on how the move went and what to try next time: praise when it
  worked (it moved the game forward / the threat was real), advice when it did not (shoot or pass earlier), good control for a dribble, and then the
  player who mattered - an opponent or a teammate, on your left or right, close or far - and why. It is built from what the camera saw, so it is an
  impression, and left / right can be a little off. The same text is in the log. Box colours: yellow = our team, magenta = the others,
  orange = a person without a team yet, white = the ball (only), dark blue = the goal.
- Colours are remembered per session name in team_colors.json (next to the exe). Delete the entry, or press "Pick shirt colour now", to change it.
- Each run writes a log to <output folder>/logs/. Send it along if something fails.
"""


def main():
    model = os.path.join(ROOT, "models", f"match_detector_{DETECTOR_VERSION}.pt")
    if not os.path.exists(model):
        sys.exit(f"missing {model}")
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onedir", "--windowed", "--name", NAME,
           "--distpath", os.path.join(ROOT, "dist"), "--workpath", os.path.join(ROOT, "build"), "--specpath", os.path.join(ROOT, "build"),
           "--paths", os.path.join(ROOT, "src"),
           "--add-data", f"{model}{os.pathsep}models",
           "--collect-all", "ultralytics", "--collect-all", "tkinterdnd2", "--collect-all", "av",
           "--exclude-module", "openvino",
           os.path.join(ROOT, "src", "app.py")]
    t0 = time.time()
    print(" ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=ROOT)
    out = os.path.join(ROOT, "dist", NAME)
    with open(os.path.join(out, "README.txt"), "w", encoding="utf-8") as f:
        f.write(README)
    size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(out) for f in fs) / 1e9
    print(f"\nbuilt {out}  ({size:.1f} GB) in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
