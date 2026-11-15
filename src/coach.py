"""
The "coach line": a few sentences at the end of every highlighted move that tell the player how it went and what to try next time.

It is a small sentence STATE MACHINE that starts from the event itself and branches as the story of the move goes on:

    HEAD (what happened: a shot on target / a missed shot / a completed or lost pass / a dribble)
      -> EVAL  success: praise (the move pushed the game forward / the threat on goal was real, strength from the event's score)
               failure: advice for next time (shoot earlier, release the pass earlier, find the free player sooner ...)
               dribble: good control and protection of the ball
      -> EXTRA (sometimes) a second thought: the next step, or one more touch when nobody is close
      -> PLAYER (usually) the player who mattered, an opponent or a teammate, with the side he was on and how close he looked, and what he
               meant for the move ("the defender on your left was right on you, so getting it away quickly was the right call")
      -> END

Every state has several phrasings and several branches; the choice is seeded by the move itself (video, frame, type), so two moves that
look alike do not read alike, and the same move reads the same on every re-render. The wording is deliberately loose and conversational.

What the sentences are built from, and how reliable that is: the event's own numbers (score, ball pace, pass length, time on the ball) and the
people the detector found around the moment of the move. Left / right is where the person stood in the picture of the chest camera, "close"
is how big he looked (a bigger box = nearer to the player), "free" = no opponent near him. All of this is approximate - a person behind the
player or out of frame is invisible, left and right can be a little off - so it is phrased as a coach's impression, never as a measurement.
The outcome of a shot is the pipeline's own label (SHOT_ON_TARGET is the default when the flight is not tracked, see AUDIO_SHOT_OUTCOME_DEFAULT).
"""
import random

import numpy as np
import pandas as pd

WINDOW = 6                  # frames either side of the moment of the move that are searched for people and the ball
LEFT_X, RIGHT_X = 0.38, 0.62      # a person left of / right of these fractions of the picture width is "on your left / right"
ON_YOU, CLOSE, STEPS = 0.45, 0.28, 0.15     # box height / picture height: right on you / close / a few steps away / further off
NEAR_BALL_HEIGHTS = 1.0     # an opponent is "on the ball" when the ball is within this many of his heights of the middle of his box
FREE_HEIGHTS = 0.8          # a teammate is "free" when no opponent's box centre is within this many of the teammate's heights of his own
HIGH_SCORE, LOW_SCORE = 60, 40
PASS_SOFT, PASS_HARD = 0.41, 0.96           # ball speed (ball-widths per frame): 25th / 75th percentile of all our passes
PASS_SHORT, PASS_LONG = 170, 480            # pixels of the 960 px wide detection frame
MAX_CHARS = 330


def _num(row, key, default=None):
    v = row.get(key, default)
    return default if v is None or pd.isna(v) else v


# ---------------------------------------------------------------- the scene: who was around
def _side(x_frac):
    return "left" if x_frac < LEFT_X else "right" if x_frac > RIGHT_X else "ahead"


def _dist(h_frac):
    return "on" if h_frac >= ON_YOU else "close" if h_frac >= CLOSE else "steps" if h_frac >= STEPS else "far"


def _people(raw, t, ball_xy=None):
    """The labelled people (team 0 = ours, 1 = opponent) in the frame nearest to `t` that has any: a list of dicts, nearest to the player first."""
    if raw is None or "team_cluster" not in raw.columns:
        return []
    w = raw[(raw["frame"] >= t - WINDOW) & (raw["frame"] <= t + WINDOW) & (raw["type"] == "person") & raw["team_cluster"].notna()]
    if w.empty:
        return []
    fr = w["frame"].iloc[(w["frame"] - t).abs().argmin()]
    g = w[w["frame"] == fr]
    out = []
    for _, p in g.iterrows():
        W, H = max(float(p["screen_w"]), 1.0), max(float(p["screen_h"]), 1.0)
        d = {"team": int(p["team_cluster"]), "x": float(p["x_center"]), "y": float(p["y_center"]), "h": float(p["height"]),
             "side": _side(float(p["x_center"]) / W), "dist": _dist(float(p["height"]) / H), "size": float(p["height"]) / H}
        if ball_xy is not None:
            d["ball_d"] = float(np.hypot(d["x"] - ball_xy[0], d["y"] - ball_xy[1])) / max(d["h"], 1.0)
        out.append(d)
    opps = [p for p in out if p["team"] == 1]
    for p in out:
        p["free"] = p["team"] == 0 and (not opps or min(np.hypot(o["x"] - p["x"], o["y"] - p["y"]) / max(p["h"], 1.0) for o in opps) > FREE_HEIGHTS)
    return sorted(out, key=lambda p: -p["size"])


def _ball_at(raw, t, before_only=True):
    if raw is None:
        return None
    b = raw[(raw["type"] == "sports ball") & (raw["frame"] >= t - WINDOW) & (raw["frame"] <= (t + 2 if before_only else t + WINDOW))]
    return (float(b["x_center"].median()), float(b["y_center"].median())) if len(b) else None


def _goal_side(raw, t):
    if raw is None:
        return None
    g = raw[(raw["type"] == "goal") & (raw["frame"] >= t - 20) & (raw["frame"] <= t + 40)]
    return _side(float(g["x_center"].median()) / max(float(g["screen_w"].iloc[0]), 1.0)) if len(g) else None


def scene(row, raw):
    """Everything the sentences are made of, as small plain labels (None = unknown)."""
    et = str(row["event_type"])
    kind = et.split("_")[0].lower()
    t = int(row["start_frame"])
    f = {"move": et.lower(), "kind": kind, "ok": None, "strength": None, "pace": None, "length": None, "hold_s": None,
         "opp": None, "mate": None, "receiver": None, "goal": None, "pressured": None}
    score = _num(row, "score")
    if score is not None:
        f["strength"] = "high" if score >= HIGH_SCORE else "low" if score < LOW_SCORE else "mid"
    if kind == "shot":
        f["ok"] = et == "SHOT_ON_TARGET"
        f["goal"] = _goal_side(raw, t)
    elif kind == "pass":
        f["ok"] = et == "PASS_SUCCESS"
        speed, length = _num(row, "speed"), _num(row, "length_px")
        if speed is not None:
            f["pace"] = "soft" if speed < PASS_SOFT else "hard" if speed > PASS_HARD else "normal"
        if length is not None:
            f["length"] = "short" if length < PASS_SHORT else "long" if length > PASS_LONG else "medium"
    else:
        f["hold_s"] = round((_num(row, "duration_frames", 0) or 0) / 30.0, 1)
        t = int(t + (_num(row, "duration_frames", 0) or 0) // 2)        # a dribble: look at the middle of the carry
    ball = _ball_at(raw, t)
    people = _people(raw, t, ball)
    opps, mates = [p for p in people if p["team"] == 1], [p for p in people if p["team"] == 0]
    on_ball = lambda p: p.get("ball_d", 9.0) <= NEAR_BALL_HEIGHTS or p["dist"] in ("on", "close")   # noqa: E731
    f["opp"] = opps[0] if opps else None
    f["mate"] = next((p for p in mates if p["free"]), mates[0] if mates else None)
    f["pressured"] = None if not people else any(on_ball(o) for o in opps)
    if kind == "pass":      # who the ball was meant for / ended with: the person nearest to the ball where the pass ended
        end = int(_num(row, "end_frame", t + 15) or t + 15)
        eb = _ball_at(raw, end, before_only=False)
        if eb is not None:
            ends = _people(raw, end, eb)
            ends = [p for p in ends if p.get("ball_d", 9.0) <= 2.0 and p["team"] == (0 if f["ok"] else 1)]
            if ends:
                f["receiver"] = min(ends, key=lambda p: p["ball_d"])
    return f


def scene_text(f):
    """The scene as one line of text (goes to the log next to the sentences)."""
    def person(p):
        return None if p is None else f"{p['side']}/{p['dist']}" + ("/free" if p.get("free") else "")
    parts = [f"move={f['move']}", f"strength={f['strength']}", f"pace={f['pace']}", f"length={f['length']}", f"hold_s={f['hold_s']}",
             f"goal={f['goal']}", f"pressured={f['pressured']}", f"opp={person(f['opp'])}", f"mate={person(f['mate'])}", f"receiver={person(f['receiver'])}"]
    return " ".join(p for p in parts if not p.endswith("=None"))


# ---------------------------------------------------------------- words
SIDE = {"left": "on your left", "right": "on your right", "ahead": "ahead of you"}
DIST = {"on": "very close", "close": "close", "steps": "a few steps away", "far": "some way off"}


def _slots(f, p=None):
    s = {"hold": f["hold_s"], "goal": SIDE.get(f["goal"], "ahead of you")}
    if p is not None:
        s["side"], s["dist"] = SIDE[p["side"]], DIST[p["dist"]]
    return s


# ---------------------------------------------------------------- the state machine
# MACHINE[state] = list of options; an option fires when its `when(f)` is true. `say` = phrasings (or a function of f returning them),
# `next` = [(state, probability), ...] (what follows; the rest of the probability = END). Clauses start with their own connector.
def _always(f):
    return True


HEAD = [
    dict(when=lambda f: f["kind"] == "shot" and f["ok"], say=[
        "That was a shot on target", "You got a shot away and it was on target", "A shot that tested the keeper",
        "You went for goal, and the shot was on target"]),
    dict(when=lambda f: f["kind"] == "shot" and f["ok"] and f["goal"], say=[
        "You shot at the goal {goal}, and it was on target", "With the goal {goal}, you let fly and the shot was on target"]),
    dict(when=lambda f: f["kind"] == "shot" and not f["ok"], say=[
        "The shot missed the target", "That attempt on goal went off target", "You tried your luck, but the shot was off target"]),
    dict(when=lambda f: f["kind"] == "pass" and f["ok"], say=[
        "That pass found its man", "A completed pass", "The ball reached your teammate", "Nice ball, it got to its target"]),
    dict(when=lambda f: f["kind"] == "pass" and f["ok"] and f["length"] == "long", say=[
        "That was a long pass and it arrived", "A long ball that found a teammate"]),
    dict(when=lambda f: f["kind"] == "pass" and not f["ok"], say=[
        "That pass did not get through", "The ball ended up with the other team", "A pass that never reached your teammate", "That ball was cut out"]),
    dict(when=lambda f: f["kind"] == "dribble" and (f["hold_s"] or 0) >= 1.0, say=[
        "You carried the ball for {hold} s", "A good spell on the ball, {hold} s of it", "You held on to the ball for {hold} s",
        "A neat carry of {hold} s with the ball"]),
    dict(when=lambda f: f["kind"] == "dribble" and (f["hold_s"] or 0) < 1.0, say=[
        "A quick carry with the ball", "A brisk touch-and-go with the ball", "You took the ball on for a moment", "A short run with the ball at your feet",
        "You nudged the ball forward and kept it"]),
]

EVAL_PRAISE = {
    "high": [" - it pushed the game forward and the threat on goal was real.",
             " - that is the kind of move that moves the game on and makes things dangerous for them.",
             " - well done, it lifted the tempo and created a real threat."],
    "mid": [" - a useful move that kept the attack going.", " - it kept the play moving the right way.",
            " - simple, but it moved the team up the pitch."],
    "low": [" - nothing flashy, but it kept the ball with your team.", " - a modest move that kept things under control.",
            " - not a big moment, but it kept the move alive.", " - simple and sensible, the team still has the ball.",
            " - small step, but those are the ones that build an attack."],
}
EVAL_SHOT_PRAISE = {
    "high": [" - a proper chance, the kind that puts the other side under pressure.", " - a real threat to their goal, keep taking those."],
    "mid": [" - the right instinct, you made them worry about the goal.", " - good to see you test them."],
    "low": [" - it was not the sharpest, but the intent to shoot was right.", " - you tried, and that is how chances turn into goals."],
}
EVAL = [
    dict(when=lambda f: f["kind"] == "shot" and f["ok"], say=lambda f: EVAL_SHOT_PRAISE[f["strength"] or "mid"],
         next=[("EXTRA", 0.4), ("PLAYER", 0.58)]),
    dict(when=lambda f: f["kind"] == "pass" and f["ok"], say=lambda f: EVAL_PRAISE[f["strength"] or "mid"],
         next=[("EXTRA", 0.3), ("PLAYER", 0.65)]),
    dict(when=lambda f: f["kind"] == "shot" and not f["ok"], say=[
        " - next time shoot a touch earlier, before the defence has time to settle.",
        " - try to pull the trigger a moment sooner, the first chance is often the best one.",
        " - pick your corner before the ball arrives and strike it earlier."],
         next=[("EXTRA", 0.4), ("PLAYER", 0.58)]),
    dict(when=lambda f: f["kind"] == "pass" and not f["ok"], say=[
        " - next time release it a little earlier, before the lane closes.",
        " - try to spot the free player sooner and pass it earlier.",
        " - a quicker decision next time would have kept the ball."],
         next=[("EXTRA", 0.5), ("PLAYER", 0.5)]),
    dict(when=lambda f: f["kind"] == "dribble" and f["pressured"], say=[
        " - you protected the ball well even with a defender around you.", " - strong control under pressure, and you kept it.",
        " - good close control, and your body kept the ball safe.", " - brave on the ball with a man around you, and it paid off.",
        " - you shielded it nicely and did not panic."],
         next=[("EXTRA", 0.35), ("PLAYER", 0.62)]),
    dict(when=lambda f: f["kind"] == "dribble" and not f["pressured"], say=[
        " - your control of the ball was good and you kept it safe.", " - good close control, you looked comfortable on it.",
        " - nice and tidy, the ball stayed close to you.", " - the first touch was sharp and you never lost it.",
        " - you kept it glued to your feet, good stuff.", " - calm and in control, exactly how it should look."],
         next=[("EXTRA", 0.45), ("PLAYER", 0.5)]),
]

EXTRA = [
    dict(when=lambda f: f["kind"] == "dribble" and (f["hold_s"] or 0) >= 1.5 and not f["pressured"], say=[
        "The next step is to look up earlier - a pass or a shot may have been on.", "With that much room, lift your head sooner and look for the pass.",
        "When nobody is near, that is the moment to scan for the forward option."], next=[("PLAYER", 0.92)]),
    dict(when=lambda f: f["kind"] == "dribble", say=[
        "Keep the touches small and the head up between them.", "Next time try to release it a beat earlier, while the defence is still moving."],
         next=[("PLAYER", 0.92)]),
    dict(when=lambda f: f["kind"] == "shot" and not f["ok"] and f["pressured"] is False, say=[
        "Nobody seemed close, so one more touch to set the ball would have been fine too.",
        "You had time here - a touch to get it out of your feet, then shoot."], next=[("PLAYER", 0.92)]),
    dict(when=lambda f: f["kind"] == "shot", say=[
        "Keep your head still through the strike and follow through towards the target.", "Hit it with the laces and pick the spot early."],
         next=[("PLAYER", 0.92)]),
    dict(when=lambda f: f["kind"] == "pass" and f["ok"], say=[
        "Keep doing that, and try it a touch earlier next time.", "Good habit - play it first time when you can.",
        "Passes like this are what keep the move flowing."], next=[("PLAYER", 0.92)]),
    dict(when=lambda f: f["kind"] == "pass" and f["pace"] == "hard", say=[
        "It also came in fast; a bit less pace makes it easier to control.", "Slightly less pace and your teammate can take it in his stride."],
         next=[("PLAYER", 0.92)]),
    dict(when=lambda f: f["kind"] == "pass" and f["pace"] == "soft", say=[
        "It was a little soft too - a firmer ball gives defenders less time to step in.", "Add some pace and the defender has no chance to read it."],
         next=[("PLAYER", 0.92)]),
    dict(when=lambda f: f["kind"] == "pass" and f["length"] == "long", say=[
        "Long balls are a gamble - a shorter option would have been safer.", "A shorter pass keeps the ball more often than a long one."],
         next=[("PLAYER", 0.92)]),
    dict(when=_always, say=["Keep scanning before every touch.", "Keep your head up and keep looking around."], next=[("PLAYER", 0.92)]),
]


def _player_options(f):
    """What the PLAYER state can say, as (phrasing, the person it is about), depending on who was around and how the move went."""
    out = []
    opp, mate, rec = f["opp"], f["mate"], f["receiver"]
    kind, ok = f["kind"], f["ok"]

    def add(person, *phrasings):
        out.extend((ph, person) for ph in phrasings)

    if kind == "pass" and rec is not None and ok:
        if rec.get("free"):
            add(rec, "Your teammate {side} was free, a good place to put it.", "The man {side} was in space - the right target.")
        else:
            add(rec, "Your teammate {side} had someone near him, so getting it there took some nerve.",
                "The receiver {side} was marked and still got the ball - tidy.")
    if kind == "pass" and rec is not None and not ok:
        add(rec, "The opponent {side} read it - a quicker or firmer ball would have beaten him.", "The defender {side} was waiting for that pass.",
            "That defender {side} stepped in, so the lane was not really open.")
    if opp is not None and opp["dist"] in ("on", "close"):
        if kind == "shot" and ok:
            add(opp, "The defender {side} was {dist}, so getting the shot off quickly was exactly right.",
                "With a defender {dist} {side}, there was no time to waste, and you used it well.")
        if kind == "shot" and not ok:
            add(opp, "The defender {side} was {dist}, which probably rushed the shot - a touch earlier would have given you more room.",
                "A defender was {dist} {side}, so I understand the hurry; shooting just before he arrives is the trick.")
        if kind == "pass" and ok:
            add(opp, "The opponent {side} was {dist}, which makes the completed pass even better.",
                "A defender {side} was {dist} and the ball still got away from him.")
        if kind == "pass" and not ok:
            add(opp, "An opponent was {dist} {side}, so the simple, safe pass was the one to look for.",
                "With a defender {dist} {side}, the first option is the safest one.")
        if kind == "dribble":
            add(opp, "The opponent {side} was {dist} and still could not get the ball off you.",
                "A defender {side} tried to close you down and you kept the ball.", "You had a defender {dist} {side} and stayed calm - good.")
    elif opp is not None:
        add(opp, "The nearest opponent was {dist} {side}, so you had a little time on the ball.",
            "The defender {side} was {dist} - there was room to take a touch.")
    if mate is not None:
        if mate.get("free"):
            if kind == "shot":
                add(mate, "Your teammate {side} was free, so a pass was the other option there.",
                    "Keep an eye on the free man {side}: sometimes the pass is the better shot.")
            if kind == "pass" and not ok:
                add(mate, "Your teammate {side} was open - he was the better target.", "The man {side} was free, a quick look that way and the pass is on.")
            if kind == "dribble":
                add(mate, "Your teammate {side} was open if you wanted to release the ball.",
                    "A free teammate {side} - a short pass there would have kept the defence guessing.")
            if kind == "pass" and ok and rec is None:
                add(mate, "Another teammate {side} was free too, so you had options.")
        else:
            if kind == "dribble":
                add(mate, "Your teammate {side} was marked, so keeping the ball yourself was the sensible choice.")
            if kind == "shot":
                add(mate, "Your teammate {side} was marked, so shooting was a fair call.")
    return out


def _hedge(rng, text):
    """Now and then the sentence is spoken as an impression ('it looks like ...'), which is what it is."""
    first = text.split(" ", 1)[0]
    if rng.random() < 0.3 and first in ("The", "Your", "A", "An", "Another", "That"):
        lead = rng.choice(["It looks like "] if first in ("A", "An") else ["It looks like ", "I think ", "As I see it, "])
        return lead + text[0].lower() + text[1:]
    return text


def compose(f, rng):
    """Walk the state machine; returns the finished text."""
    options = [o for o in HEAD if o["when"](f)]
    # the most specific phrasing (listed last, e.g. the one that mentions the goal's side or a long pass) wins half of the time, else any that fits
    head = options[-1] if len(options) > 1 and rng.random() < 0.5 else rng.choice(options)
    text = rng.choice(head["say"]).format(**_slots(f))
    state_opts = [o for o in EVAL if o["when"](f)]
    if not state_opts:
        return text + "."
    ev = state_opts[0]
    ev_say = ev["say"](f) if callable(ev["say"]) else ev["say"]
    text += rng.choice(ev_say)
    nxt = ev["next"]
    visited = set()
    state = _pick(rng, nxt)
    while state in ("EXTRA", "PLAYER") and state not in visited:
        visited.add(state)
        if state == "EXTRA":
            opts = [o for o in EXTRA if o["when"](f)]
            o = rng.choice(opts[:2]) if len(opts) > 1 else opts[0]      # the first, most specific options come first
            text += " " + rng.choice(o["say"])
            state = _pick(rng, o["next"])
        else:
            cands = _player_options(f)
            if cands:
                phrasing, person = rng.choice(cands)
                text += " " + _hedge(rng, phrasing.format(**_slots(f, person)))
            state = "END"
    return text if len(text) <= MAX_CHARS else text[:text.rfind(".", 0, MAX_CHARS) + 1]


def _pick(rng, nxt):
    r, acc = rng.random(), 0.0
    for state, p in nxt:
        acc += p
        if r < acc:
            return state
    return "END"


def coach_line(row, raw, seed_key=""):
    """(text, scene text) for one event row. `raw` = the video's detections table (with team_cluster), `seed_key` = e.g. the video's name."""
    f = scene(row, raw)
    rng = random.Random(f"{seed_key}|{int(row['start_frame'])}|{row['event_type']}")
    return compose(f, rng), scene_text(f)
