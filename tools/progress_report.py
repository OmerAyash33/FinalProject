"""
Progress report: where the detections stand NOW and whether we PROGRESSED since the start, on the same labels.

Sections
  1. Progress on the user's labels (calibration videos and the 4 held-out January videos), three runs of tools/benchmark.py on the SAME 227 labels:
     START (old settings), BEFORE (settings of 2026-11-08: pass calibration + audio kick cue), NOW (+ stage B shot self-check). Recall, class accuracy,
     events, precision lower bound, and the Wilson 95% interval of the held-out recall (what to expect on a video nobody has tuned on).
  2. What would reach the highlight reel: the top-3 shots per kind and session (select_events) with the self-check OFF and ON, real or not.
  3. The videos nobody has seen (pipeline_outputs/unseen_day_<v>, unseen_night_<v>): events per minute by family compared with the labelled day / night
     sessions, the shots by origin (ball track / audio+goal / self-check) and their confidence. No labels exist for them: counts and rates only.
  4. A simulation of what may happen on the next unseen session (intervals, not promises).

Usage:  .venv/Scripts/python.exe tools/progress_report.py [--out report.md]   (needs the benchmark rows of the three runs in benchmark_history.csv)
"""
import glob
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "tools"))

import torch  # noqa: F401,E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import benchmark as B  # noqa: E402
import event_params as P  # noqa: E402
from config import DETECTOR_VERSION  # noqa: E402
from event_scoring import score_events, select_events  # noqa: E402

DAY_SESSIONS = {"2026-01-30", "2026-05-01_morning"}          # labelled sessions shot in daylight (mean frame brightness ~118)
NIGHT_SESSIONS = {"2026-05-05_evening", "2026-07-28"}        # labelled sessions under floodlights (~54)
FAMILY = B.FAMILY_OF_EVENT


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def pick_run(hist, startswith=None, contains=None):
    h = hist
    if startswith:
        h = h[h["note"].str.startswith(startswith)]
    if contains:
        h = h[h["note"].str.contains(contains, regex=False)]
    if h.empty:
        return None
    last = h.sort_values("time").iloc[-1]["time"]
    return h[h["time"] == last]


def table_progress(hist):
    runs = {"START (old settings)": pick_run(hist, contains="OLD settings: pre-calibration passes, no audio cue, no shot self-check"),
            "BEFORE (2026-11-08 settings)": pick_run(hist, startswith="after full labelling 2026-11-14"),
            "NOW (+ shot self-check)": pick_run(hist, startswith="shot self-check stage B added")}
    lines = []
    for split, title in (("calib", "Calibration videos (31 videos, 174 min)"), ("heldout", "HELD-OUT January videos 2/4/6/8 (36 min, never tuned on)")):
        lines += [f"### {title}", "", "| family | run | labels found | recall | class acc | events | precision (lower bound) | events/min |", "|---|---|---|---|---|---|---|---|"]
        for fam, name in (("PASS", "passes"), ("SHOT", "shots"), ("DRIB", "dribbles")):
            for rn, df in runs.items():
                if df is None:
                    continue
                r = df[(df["split"] == split) & (df["family"] == fam)]
                if r.empty:
                    continue
                r = r.iloc[0]
                found = int(round(r["recall"] * r["labels"])) if pd.notna(r["recall"]) else 0
                lo, hi = wilson(found, int(r["labels"]))
                ca = "-" if pd.isna(r["class_acc"]) else f"{r['class_acc']:.2f}"
                pr = "-" if pd.isna(r["precision_lower_bound"]) else f"{r['precision_lower_bound']:.2f}"
                lines.append(f"| {name} | {rn} | {found} of {int(r['labels'])} | {r['recall']:.2f} ({lo:.2f}-{hi:.2f}) | {ca} | {int(r['events'])} | {pr} | {r['events_per_min']:.2f} |")
        lines.append("")
    return lines, runs


def verdicts(runs):
    """One line per family and split: did the recall move from START and from BEFORE to NOW, and is the move more than noise (Wilson intervals)?"""
    now, start, before = runs["NOW (+ shot self-check)"], runs["START (old settings)"], runs["BEFORE (2026-11-08 settings)"]
    if now is None or start is None or before is None:
        return []
    lines = ["### Did we progress? (recall, same labels)", "", "| family | split | START | BEFORE | NOW | verdict |", "|---|---|---|---|---|---|"]
    for fam, name in (("PASS", "passes"), ("SHOT", "shots"), ("DRIB", "dribbles")):
        for split in ("calib", "heldout"):
            def get(df):
                r = df[(df["split"] == split) & (df["family"] == fam)].iloc[0]
                return int(round(r["recall"] * r["labels"])), int(r["labels"])
            (ks, n), (kb, _), (kn, _) = get(start), get(before), get(now)
            from scipy.stats import fisher_exact
            pval = float(fisher_exact([[kn, n - kn], [ks, n - ks]], alternative="greater")[1])   # same labels in both runs; Fisher is conservative for that
            if kn == ks:
                verdict = "no change"
            elif kn < ks:
                verdict = "WORSE"
            elif pval < 0.05:
                verdict = f"clear progress (p = {pval:.3f})"
            elif pval < 0.20:
                verdict = f"probably progress, not proven (p = {pval:.2f})"
            else:
                verdict = f"up, but within noise (p = {pval:.2f})"
            lines.append(f"| {name} | {split} | {ks}/{n} | {kb}/{n} | {kn}/{n} | {verdict} |")
    return lines + [""]


def events_of(session, video):
    ev, n = B.stage2_events(session, video)
    return ev, n


def reel_simulation(labelled):
    """Top-3 shots per kind and session as select_events would pick them (SELECTION_SCOPE session), self-check OFF vs ON."""
    lab = pd.read_csv(B.LABELS, encoding="utf-8-sig")
    shot_lab = lab[lab.label.isin(["SHOT_ON_TARGET", "SHOT_OFF_TARGET", "GOAL"])]
    out = {}
    for enabled in (False, True):
        P.SHOT_CHECK_ENABLED = enabled
        rows = []
        for (session, video) in labelled:
            ev, _ = events_of(session, video)
            if not len(ev):
                continue
            ev = ev[ev.event_type.isin(["SHOT_ON_TARGET", "SHOT_OFF_TARGET"])]
            if not len(ev):
                continue
            sl = shot_lab[shot_lab.video == video].frame.values
            for e in score_events(ev.to_dict("records")):
                e["video"], e["session"] = video, session
                e["hit"] = bool(len(sl)) and bool(((sl >= e["start_frame"] - 15) & (sl <= e["end_frame"] + 90)).any())
                e["split"] = "heldout" if video in B.HELDOUT_VIDEOS else "calib"
                rows.append(e)
        D = pd.DataFrame(rows)
        for split in ("calib", "heldout"):
            n_sel = hits = 0
            for session, g in D[D["split"] == split].groupby("session"):
                g = g.copy()
                g["pool"] = session
                sel = select_events(g)                                   # keeps the input columns, `hit` included
                n_sel += len(sel)
                hits += int(sel["hit"].sum())
            out[(enabled, split)] = (n_sel, hits)
    P.SHOT_CHECK_ENABLED = True
    return out


def unseen_summary():
    lines = []
    for name, label in (("unseen_day", "day"), ("unseen_night", "night")):
        d = os.path.join(PROJECT_ROOT, "pipeline_outputs", f"{name}_{DETECTOR_VERSION}")
        files = sorted(f for f in glob.glob(os.path.join(d, "*_events.csv")) if not os.path.basename(f).startswith("selected"))   # not selected_events.csv
        if not files:
            lines.append(f"- {label}: no events yet")
            continue
        ev, minutes = [], 0.0
        for f in files:
            e = pd.read_csv(f)
            e.insert(0, "video", os.path.basename(f)[:-len("_events.csv")])
            ev.append(e)
            raw = pd.read_csv(f.replace("_events.csv", "_data.csv"), usecols=["frame"])
            minutes += raw["frame"].max() / 1800.0
        E = pd.concat(ev, ignore_index=True)
        E["family"] = E["event_type"].map(FAMILY)
        rate = (E.groupby("family").size() / minutes).round(2).to_dict()
        shots = E[E["family"] == "SHOT"]
        lines.append(f"- **{label}**: {len(files)} videos, {minutes:.0f} min, {len(E)} events -> per minute {rate}; shots by origin "
                     f"{shots['shot_origin'].value_counts().to_dict() if len(shots) else {}}, median shot confidence "
                     f"{shots['shot_confidence'].median():.2f}" if len(shots) else f"- **{label}**: {len(files)} videos, {minutes:.0f} min, {len(E)} events -> per minute {rate}; no shots")
    return lines


def labelled_rates(labelled):
    rows = []
    for (session, video) in labelled:
        ev, n = events_of(session, video)
        if len(ev):
            ev = ev.copy()
            ev["family"] = ev["event_type"].map(FAMILY)
            rows.append((session, n / 1800.0, ev["family"].value_counts().to_dict()))
        else:
            rows.append((session, n / 1800.0, {}))
    res = {}
    for kind, sess in (("day", DAY_SESSIONS), ("night", NIGHT_SESSIONS)):
        minutes = sum(m for s, m, _ in rows if s in sess)
        counts = {}
        for s, _, c in rows:
            if s in sess:
                for k, v in c.items():
                    counts[k] = counts.get(k, 0) + v
        res[kind] = (float(minutes), {k: round(float(v) / minutes, 2) for k, v in counts.items()})
    return res


def main():
    out_path = os.path.join(PROJECT_ROOT, "pipeline_outputs", "_analysis", f"progress_report_{time.strftime('%Y-%m-%d')}.md")
    if "--out" in sys.argv:
        out_path = sys.argv[sys.argv.index("--out") + 1]
    hist = pd.read_csv(B.HISTORY)
    lab = pd.read_csv(B.LABELS, encoding="utf-8-sig")
    labelled = list(lab.groupby(["session", "video"], sort=False).groups.keys())
    L = [f"# Progress report {time.strftime('%Y-%m-%d %H:%M')}", "",
         f"Detector {DETECTOR_VERSION}; {len(lab)} labels of the user in {lab['video'].nunique()} videos. Same labels for every run below, so the rows are comparable.", "",
         "## 1. Progress on the user's labels", ""]
    t, runs = table_progress(hist)
    L += t
    L += verdicts(runs)
    L += ["Reading it: recall = share of the user's labels that an event of the same family matches (event start - 15 <= label <= event end + 90 frames); "
          "precision is only a lower bound (the user may have missed some real events); the interval after the held-out recall is the Wilson 95% interval: with 5 shots "
          "or 20 passes it is wide, so a difference of one or two events is NOT proof of progress.", ""]
    print("running the reel simulation (2 x Stage 2 on 35 videos)...", flush=True)
    labelled_pairs = labelled
    sim = reel_simulation(labelled_pairs)
    L += ["## 2. What would reach the highlight reel (top-3 shots per kind and session)", "",
          "| split | self-check OFF: shots in reel / real | self-check ON: shots in reel / real |", "|---|---|---|"]
    for split in ("calib", "heldout"):
        a, b = sim[(False, split)], sim[(True, split)]
        L.append(f"| {split} | {a[0]} / {a[1]} | {b[0]} / {b[1]} |")
    L += ["", "\"Real\" = a label of the user within the matching window. These are single-digit counts.", ""]
    print("rates of the labelled day / night sessions...", flush=True)
    rates = labelled_rates(labelled_pairs)
    L += ["## 3. Videos nobody has seen (no labels: counts and rates only)", ""]
    L += unseen_summary()
    L += ["", "For comparison, the labelled sessions with the current rules (events per minute): "
          + "; ".join(f"{k}: {v[1]} ({v[0]:.0f} min)" for k, v in rates.items()), ""]
    manual = os.path.join(PROJECT_ROOT, "pipeline_outputs", "_analysis", "unseen_shots_manual_check.md")
    if os.path.exists(manual):
        L += ["### Manual check of the shots found in the unseen videos", ""] + open(manual, encoding="utf-8").read().splitlines() + [""]
    held = runs["NOW (+ shot self-check)"]
    L += ["## 4. What to expect on the next unseen session (simulation from the held-out numbers; intervals, not promises)", ""]
    if held is not None:
        for fam, name in (("PASS", "passes"), ("SHOT", "shots"), ("DRIB", "dribbles")):
            r = held[(held["split"] == "heldout") & (held["family"] == fam)].iloc[0]
            found = int(round(r["recall"] * r["labels"]))
            lo, hi = wilson(found, int(r["labels"]))
            L.append(f"- {name}: recall about {r['recall']:.0%} (plausible range {lo:.0%}-{hi:.0%}), about {r['events_per_min']:.2f} events per minute, "
                     f"at least {r['precision_lower_bound']:.0%} of them real" if pd.notna(r["precision_lower_bound"]) else f"- {name}: no events")
    L += ["- shots, the 13 shots found in the unseen videos: the user says all 13 are real kicks (my first picture-only check said 15-25%, withdrawn: it did not use the sound); "
          "the open question there is recall (how many kicks are missed), which no one can measure without labels",
          "- the final blind session: expect these numbers or slightly worse; the benchmark's 'precision' of shots is only a lower bound, 25 of 36 unmatched shot events in the labelled videos have no label at all and 22 of them a loud click", ""]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print("\nwritten:", out_path)


if __name__ == "__main__":
    main()
