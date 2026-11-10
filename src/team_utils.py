"""
Team identity helpers shared by Stage 1 (live preview colours) and Stage 2 (the real assignment).

Convention used everywhere: team 0 = Team A = OUR team (drawn yellow), team 1 = Team B = the opponents
(drawn magenta). Which colour is ours comes, in order of preference, from
  1. the user's own pick for this session (team_colors.json, written by team_color_picker.py): the reference colour is the
     median of the shirts the user clicked, TeamModel.from_reference;
  2. the per-session light/dark rule: KMeans only says "cluster 0 / cluster 1" in arbitrary order, so the clusters are told
     apart by BRIGHTNESS and event_params.OUR_SHADE_* says whether ours is the lighter or the darker one.
build_team_model() applies this choice.
"""
import os

import cv2
import numpy as np

import event_params as P
from config import DETECTION_COLORS


def our_shade(session_name):
    """"light" or "dark": which jersey shade is our team in this session (see event_params)."""
    return P.OUR_SHADE_BY_SESSION.get(session_name, P.OUR_SHADE_DEFAULT)


def session_of(path):
    """Session folder name of a raw video or of a pipeline output file ('..._v2' version suffix stripped)."""
    name = os.path.basename(os.path.dirname(os.path.abspath(path)))
    for ver in ("_v1", "_v2"):
        if name.endswith(ver):
            return name[: -len(ver)]
    return name


def brightness(bgr):
    """Perceived brightness (0-255) of a B,G,R colour (Rec.601 luma)."""
    b, g, r = np.asarray(bgr, dtype=float)[..., 0], np.asarray(bgr, dtype=float)[..., 1], np.asarray(bgr, dtype=float)[..., 2]
    return 0.114 * b + 0.587 * g + 0.299 * r


def cluster_to_team(centers_bgr, shade):
    """
    Array `m` with m[cluster] = team (0 = ours / Team A, 1 = opponents / Team B).
    The cluster whose centre is brighter is the "light" one; `shade` says whether our team is the light or the dark one.
    """
    luma = brightness(centers_bgr)
    light, dark = int(np.argmax(luma)), int(np.argmin(luma))
    ours = light if shade == "light" else dark
    m = np.ones(len(centers_bgr), dtype=int)
    m[ours] = 0
    return m


def team_features(colors_bgr):
    """(n, 3) B,G,R colours -> (n, 3) comparison features: saturation-weighted hue as a point on a circle, plus half-weight brightness."""
    bgr = np.clip(np.atleast_2d(np.asarray(colors_bgr, dtype=float)), 0, 255).astype(np.uint8).reshape(-1, 1, 3)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
    hue = hsv[:, 0] * 2.0 * np.pi / 180.0
    return np.c_[hsv[:, 1] * np.cos(hue), hsv[:, 1] * np.sin(hue), P.TEAM_FEATURE_V_WEIGHT * hsv[:, 2]]


class TeamModel:
    """
    Two jersey colours learned by KMeans (on team_features), with the light/dark rule applied: team 0 = OUR team (the lighter or
    the darker cluster, per `shade`), team 1 = the opponents. `colors_bgr` is one B,G,R row per tracked player (median chest colour).
    """

    def __init__(self, colors_bgr, shade, random_state=42):
        from sklearn.cluster import KMeans  # lazy: pandas/torch import order matters at start-up
        colors = np.asarray(colors_bgr, dtype=float)
        km = KMeans(n_clusters=2, random_state=random_state, n_init="auto").fit(team_features(colors))
        self.feat_centers = km.cluster_centers_
        self.bgr_centers = np.array([colors[km.labels_ == k].mean(axis=0) for k in range(2)])
        self.shade = shade
        self.to_team = cluster_to_team(self.bgr_centers, shade)     # cluster index -> 0 (ours) / 1 (opponents)
        self.ours_cluster = int(np.where(self.to_team == 0)[0][0])
        self.theirs_cluster = int(np.where(self.to_team == 1)[0][0])

    @classmethod
    def from_reference(cls, colors_bgr, ours_bgr, theirs_bgr=None):
        """
        Model whose "ours" colour is the one the USER picked: the median (in team_features space) of the clicked shirts `ours_bgr`.
        "Theirs" is the median of the shirts the user marked as opponents, or - when they only picked their own colour - the
        KMeans centre (of all `colors_bgr`) that lies farthest from ours. Works for any colour (yellow, blue, red ...): there is no
        light/dark rule involved.
        """
        from sklearn.cluster import KMeans  # lazy, see __init__
        self = cls.__new__(cls)
        ours = np.atleast_2d(np.asarray(ours_bgr, dtype=float))
        f_ours = np.median(team_features(ours), axis=0)
        if theirs_bgr is not None and len(theirs_bgr) > 0:
            theirs = np.atleast_2d(np.asarray(theirs_bgr, dtype=float))
            f_theirs, bgr_theirs = np.median(team_features(theirs), axis=0), np.median(theirs, axis=0)
        else:
            colors = np.asarray(colors_bgr, dtype=float)
            km = KMeans(n_clusters=2, random_state=42, n_init="auto").fit(team_features(colors))
            far = int(np.argmax(np.linalg.norm(km.cluster_centers_ - f_ours, axis=1)))
            f_theirs, bgr_theirs = km.cluster_centers_[far], colors[km.labels_ == far].mean(axis=0)
        self.feat_centers = np.vstack([f_ours, f_theirs])
        self.bgr_centers = np.vstack([np.median(ours, axis=0), bgr_theirs])
        self.shade = "reference"
        self.to_team = np.array([0, 1])
        self.ours_cluster, self.theirs_cluster = 0, 1
        return self

    def prob_ours(self, colors_bgr):
        """P(our team) per colour in 0..1 from the distances to the two centres (0.5 = exactly between them)."""
        d = np.linalg.norm(team_features(colors_bgr)[:, None, :] - self.feat_centers[None, :, :], axis=2)
        d_ours, d_theirs = d[:, self.ours_cluster], d[:, self.theirs_cluster]
        return d_theirs / np.maximum(d_ours + d_theirs, 1e-6)

    def team(self, colors_bgr):
        """Array of 0 (ours) / 1 (opponent), one per colour."""
        return np.where(self.prob_ours(colors_bgr) >= 0.5, 0, 1)


def build_team_model(colors_bgr, session):
    """The team model of a session: from the user's picked colours when team_colors.json has them, else KMeans + the light/dark rule."""
    from team_colors import load_reference   # lazy: keeps team_utils importable without the json file machinery
    ref = load_reference(session)
    if ref is not None:
        return TeamModel.from_reference(colors_bgr, ref["ours"], ref["theirs"] if len(ref["theirs"]) else None)
    return TeamModel(colors_bgr, our_shade(session))


def team_color(team):
    """Box colour (BGR) for a team value: 0 -> Team A yellow, 1 -> Team B magenta, anything else (NaN, -1) -> grey."""
    if team is None or (isinstance(team, float) and np.isnan(team)):
        return DETECTION_COLORS["player_unassigned"]
    team = int(team)
    if team == 0:
        return DETECTION_COLORS["team_a"]
    if team == 1:
        return DETECTION_COLORS["team_b"]
    return DETECTION_COLORS["player_unassigned"]


class LiveTeamColorizer:
    """
    Stage 1 live preview only: gives each player box its team colour WHILE the video is still being read, before
    Stage 2's real clustering exists. It learns the two jersey colours online (a TeamModel refitted every `refit_every`
    frames on the chest colours of the larger players seen so far) and smooths each track's label over its last few
    frames. Until it has seen enough players it returns None (drawn grey). The CSV never contains these labels - Stage 2
    recomputes every team from the whole video - so a preview colour can differ slightly.
    """

    def __init__(self, shade, min_samples=60, refit_every=45, max_samples=2500, reference=None):
        self.shade = shade
        self.reference = reference   # team_colors.load_reference() dict or None
        self.min_samples = min_samples
        self.refit_every = refit_every
        self.max_samples = max_samples
        self.samples = []
        self.model = None
        self._votes = {}
        self._last_fit_frame = 0

    def observe(self, frame_no, colors_bgr):
        """Add this frame's (large enough) player chest colours to the pool and refit when it is time."""
        for c in colors_bgr:
            if len(self.samples) >= self.max_samples:
                self.samples.pop(np.random.randint(len(self.samples)))
            self.samples.append(c)
        if len(self.samples) >= self.min_samples and frame_no - self._last_fit_frame >= self.refit_every:
            samples = np.asarray(self.samples)
            self.model = (TeamModel.from_reference(samples, self.reference["ours"], self.reference["theirs"] if len(self.reference["theirs"]) else None)
                          if self.reference is not None else TeamModel(samples, self.shade, random_state=0))
            self._last_fit_frame = frame_no

    def team_for(self, track_id, color_bgr, box_height):
        """0 (ours), 1 (opponent) or None while unknown / the box is too small to read a colour from. A track's last 15 frames vote."""
        if self.model is None or box_height < P.TEAM_MIN_BOX_HEIGHT_PX:
            return None
        p = float(self.model.prob_ours(color_bgr)[0])
        if track_id == -1:   # no stable ID, nothing to smooth over: use this single frame's colour
            return 0 if p >= 0.5 else 1
        votes = self._votes.setdefault(track_id, [])
        votes.append(p)
        del votes[:-15]
        return 0 if np.mean(votes) >= 0.5 else 1
