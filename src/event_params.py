"""
EVERY tunable number of the event logic (Stage 2) and of the highlight selection, in one place.

Change a value here and the next run of Stage 2 gives different results; nothing in the code
hides a second copy. All frame counts assume 30 fps. "bw" = ball widths (the ball's own on-screen
width), used as the unit of speed so no camera calibration is needed. Positions are fractions of
the frame (x of the width, y of the height) unless the name says px (pixels of the 960-wide
frame Stage 1 works on).

Feel for the speed unit: a size-5 ball is 0.22 m wide and the video runs at 30 fps, so
    1 bw/frame = 0.22 m x 30 = 6.6 m/s = 24 km/h      0.5 bw/frame = 12 km/h      2.5 bw/frame = 59 km/h

How the starting values were chosen: tools/analyze_event_params.py measures them on the data of
the two May sessions (footage the detector was never trained on) and prints the distributions;
each block below says which measurement it came from. Re-run it after any detector change.
"""

# ===========================================================================
# 0. MATCH SETTINGS - the one thing the video cannot tell us
# ===========================================================================
# Which jersey shade is OUR team ("light" or "dark"), per session folder name. The two clusters
# found by KMeans are told apart by brightness (the lighter centre = "light"), and OUR cluster becomes
# Team A (yellow on screen), the other Team B (magenta). Pass success = the receiver is on Team A.
# UNVERIFIED GUESS - please set it per session below. On the 12 May videos the two settings simply swap the pass results:
# "dark" -> 5 completed / 13 incomplete, "light" -> 13 completed / 5 incomplete. A friendly game's completion rate is far more likely
# to be ~70% than ~28%, so "light" is the default until the real shade of each session is entered.
OUR_SHADE_DEFAULT = "light"
OUR_SHADE_BY_SESSION = {
    # "2026-05-05_evening": "light",
}

# How a player's colour is turned into a team (Stage 2, and the colours of the Stage 1 live preview).
# Measured on the May sessions: the chest crop of a small, far player is mostly grass or sky, so only players whose box is at
# least TEAM_MIN_BOX_HEIGHT_PX tall (960-wide frame) take part in the clustering; smaller ones stay orange ("no team").
# Colour is compared as (saturation*cos(hue), saturation*sin(hue), TEAM_FEATURE_V_WEIGHT*brightness) rather than raw B,G,R:
# raw BGR is dominated by brightness, so the same orange vest in sun and in shade landed in different teams.
TEAM_MIN_BOX_HEIGHT_PX = 40
# Every "_PX" value in this file is given for a frame REF_WIDTH_PX wide (what a GPU run produces). A CSV made at another width - the CPU path
# works at 480 - must be compared with the value scaled to its own frame: use px(value, screen_w). Without this a person had to be twice as
# tall (relative to the picture) to get a team on the CPU path, and pass lengths came out half as long. (Added 2026-11-16.)
REF_WIDTH_PX = 960


def px(value, screen_w):
    """A pixel value given for a REF_WIDTH_PX-wide frame, expressed in a frame `screen_w` wide (screen_w may be a pandas column)."""
    return value * screen_w / float(REF_WIDTH_PX)


TEAM_FEATURE_V_WEIGHT = 0.5
# Team colour picker (src/team_color_picker.py): a HARD limit of this many clicked players per colour (ours / opponents) ...
PICKER_MAX_PER_COLOR = 5
# ... and as soon as this many players in total have been clicked, the picker saves and closes by itself, so the app goes on to the processing stage.
PICKER_AUTO_FINISH_TOTAL = 7

# ===========================================================================
# 1. BALL TRACK CLEANING AND SPEED
# ===========================================================================
# The ball is seen in only 7-15% of frames, in runs of ~2 frames, so everything below works on
# sparse sightings.
BALL_MIN_CONF = 0.25              # Stage 1's detector threshold; confirmed best-F1 point for the ball (0.26) on the val split
BALL_MAX_SPEED_BW = 12.0          # a ball cannot move faster than this (scene-relative bw/frame): such a jump is a detector error
BALL_JUMP_NEIGHBOUR_FRAMES = 4    # a sighting is dropped as a jump if BOTH neighbours within this many frames imply an impossible speed
BALL_INTERP_MAX_GAP = 15          # fill gaps up to this many frames with a straight line (flagged as interpolated, not seen)
CAMERA_MIN_RESP = 0.15            # camera-motion estimates with a weaker correlation peak are ignored (treated as no camera motion)
SPEED_MAX_PAIR_DT = 8             # two sightings further apart than this are not used for a speed (the path between them is unknown)
SPEED_WINDOW_FRAMES = 12          # window after a trigger over which the "mean ball speed" is computed (= PASS_END_OFFSET_FRAMES)

# ===========================================================================
# 2. WEARER'S POSSESSION ("self zone") AND HOLD CLASSES
# ===========================================================================
# The ball is "held" while it sits in the lower part of the picture, close to the wearer's feet.
SELF_ZONE_ENTER_Y = 0.60          # ball centre below this fraction of the frame height -> enters the zone
SELF_ZONE_EXIT_Y = 0.50           # ... and only leaves when it rises above this (hysteresis stops flicker at the edge)
SELF_ZONE_MIN_BALL_W = 0.03       # ball narrower than this fraction of the frame width is a far-away ball, not at the feet
SELF_ZONE_X_MIN, SELF_ZONE_X_MAX = 0.03, 0.97
HOLD_CONTESTED_DIST_HEIGHTS = 0.8 # a ball whose nearest player has his feet within this many of HIS heights is that player's ball, not the wearer's
PERSON_MIN_ASPECT = 1.4           # person boxes flatter than height/width = 1.4 are the wearer's own feet/knees/arms, not players (the 8.6% flattest boxes)
HOLD_GRACE_FRAMES = 15            # the hold survives this many frames without any sighting (the ball is often missed)
HOLD_MIN_SEEN_FRAC = 0.3         # ... and the ball must be SEEN in at least this share of the hold's frames (the lowest 10% of holds sit under 0.4; the ones under 0.3 that were inspected were the wearer's shoe or a bag, not a ball)
HOLD_MIN_HITS = 3                 # a hold needs at least this many in-zone sightings (kills single false detections)

# Hold length classes -> the three dribble shades (lightest = shortest, darkest = longest).
# A hold shorter than DRIBBLE_MIN_FRAMES is a touch and produces no dribble event.
DRIBBLE_MIN_FRAMES = 12           # 0.4 s  (same as the old DRIBBLE threshold)
DRIBBLE_MEDIUM_MIN_FRAMES = 20    # from this length: medium (0.67 s). The three classes are the terciles of the 197 holds >= 12 frames in 21 videos (12 May + 9 July;
DRIBBLE_LONG_MIN_FRAMES = 33      # from this length: long (1.1 s)         May alone gave 18 / 27, July alone 22 / 39 - the pooled values sit in between). Re-derive with tools/analyze_event_params.py (section 2)
                                  # Check over all 38 videos (May + July + January, 318 holds): 19 / 32 - one frame from the values here, so these were kept.
# Opponents count as "close to the ball" for a dribble's pressure when their feet are within this many of THEIR OWN heights of it. The hold itself is only
# defined where nobody is within HOLD_CONTESTED_DIST_HEIGHTS (0.8) of the ball, so the old radius of 1.0 left a ring of 0.2 heights and 96% of dribbles had
# pressure 0. Measured 2026-11-15 on 297 holds (39 videos), share of dribbles with any opponent close: radius 1.0 -> 3%, 1.5 -> 8%, 2.0 -> 11%, 2.5 -> 15%, 3.0 -> 19%.
# 2.0 heights is about 3.5 m for an adult. Only players with a team colour count (opponents = team 1).
DRIBBLE_PRESSURE_RADIUS_HEIGHTS = 2.0

# ===========================================================================
# 3. PASS
# ===========================================================================
# A flight starts when a hold ends (or when a fast ball leaves the zone without a hold).
PASS_TRIGGER_SPEED_BW = 0.25      # scene-relative mean ball speed (bw/frame, 0.25 = 6 km/h) that counts as "the ball is travelling"; the receiver test is the real filter.
                                  # Calibrated on the user's labels 2026-11-07 (44 passes): real chest-cam passes leave the hold at 0.2-0.6 bw/f (median ~0.4), so the old 0.5 dropped 84% of them
FAST_KICK_TRIGGER_SPEED_BW = 2.5   # second way a flight can start (besides a hold ending): two sightings <=3 frames apart in the lower picture moving faster than this (a first-time kick leaves no hold)
FAST_KICK_MAX_SPEED_BW = 6.0      # ... but faster than this (143 km/h) is a detector jump, not a ball
PASS_END_OFFSET_FRAMES = 12       # the receiver is looked up this many frames after the flight starts (user's global, default 12)
PASS_SEARCH_MAX_FRAMES = 20       # if the ball is not seen in the first PASS_END_OFFSET_FRAMES, look this far for its last sighting
PASS_RECEIVER_MAX_DIST_HEIGHTS = 2.5   # the receiver's feet must be within this many player-heights of the ball (was 1.5; most missed labelled passes had a far/small receiver)
PASS_VECTOR_LOOKBACK_FRAMES = 6   # ball not seen after the kick: its direction/speed come from the sightings this many frames before the hold ended
PASS_VECTOR_MAX_PX = 480.0        # ... and the extrapolated end point is at most this far (scene px at 960 wide) from where the ball left
PASS_VECTOR_RECEIVER_DIST_HEIGHTS = 4.0  # for an extrapolated end point the nearest player may be this many of his heights away (was 2.5)
RECEIVER_FRAME_TOLERANCE = 2      # persons are looked up in the reference frame +-this many frames
TEAM_VOTE_WINDOWS = (6, 8, 10)    # the receiver's team is decided over windows of this many frames (user: 6, 8, 10)
TEAM_VOTE_MODE = "average"        # "average": mean of the three windows' team probabilities; "best": the most confident window
TEAM_MIN_MARGIN = 0.15            # |2*P(ours)-1| below this -> the receiver's colour is too close to both teams: pass dropped
UNRESOLVED_PASS = "drop"          # flight whose receiver cannot be found: "drop" (no event) or "incomplete"
PRESSURE_RADIUS_HEIGHTS = 2.0     # opponents within this many of THEIR OWN heights of the receiver's feet count as pressure (passes; 1.0 until 2026-11-15, enlarged to 2.0 at the user's
                                  # request, same as the dribble radius). For a pass that did NOT succeed the receiver is himself
                                  # an opponent and counts too (at least 1): the ball ended up where a defender was, which is exactly what made the pass hard.

# ===========================================================================
# 4. SHOT
# ===========================================================================
SHOT_SPEED_BW = 2.5               # scene-relative mean ball speed (bw/frame) above which a kick is a shot
SHOT_GOAL_LOOKBACK_FRAMES = 30    # the goal must have been seen within this many frames before the kick ...
SHOT_WINDOW_FRAMES = 60           # ... or this many after it (the ball's flight is followed for this long)
SHOT_MIN_GOAL_FRAMES = 3          # goal detections needed in that span
GOAL_CARRY_FRAMES = 30            # the last seen goal box is carried (moved with the camera) for this many frames
SHOT_MIN_SIGHTINGS = 4             # the ball must really be SEEN this many times in the shot window (two false "shots" on the May footage had 3 sightings in 60 frames)
SHOT_MAX_WIDTH_RATIO = 0.6        # a shot leaves the camera: the ball's last width must be at most this fraction of its width at the kick (a far, resting ball seen while the camera whips has ratio ~1)
SHOT_MIN_START_CONF = 0.45        # the first ball sighting of the shot window needs at least this detector confidence (the 3 false shots in the 9 January videos were a hand/shoe at the picture's corner at 0.3, or flat boxes; real balls at the feet score 0.6-0.9)
SHOT_PAIR_MAX_DT = 4              # a "flight pair" = two sightings at most this many frames apart ...
SHOT_PAIR_MIN_SPEED_BW = 1.25     # ... moving at least this fast (half of SHOT_SPEED_BW) and at most FAST_KICK_MAX_SPEED_BW (faster = detector jump) ...
SHOT_MIN_FAST_PAIRS = 2           # ... and at least this many such pairs must move in the same direction (cosine >= SHOT_PAIR_MIN_COS): a real kick leaves a consistent trail
SHOT_PAIR_MIN_COS = 0.5
SHOT_ON_TARGET_FRAMES = 5         # frames the ball must overlap the goal for an on-target shot (user's N, default 5)
SHOT_OVERLAP_MIN_FRACTION = 0.8   # "full overlap": at least this share of the ball box lies inside the goal box
SHOT_MIN_DIRECTION_COS = 0.5      # the ball must travel towards the goal: cosine between motion and ball->goal direction
SHOT_UNRESOLVED_AS = "off_target" # shot-like kick towards a visible goal that never overlaps it: "off_target" or "drop"

# ===========================================================================
# 4b. AUDIO KICK CUE (shots the ball track cannot see)   -> src/audio_utils.py
# ===========================================================================
# On the chest-cam the ball vanishes right after a real shot (blur, camera swings with the body), so SHOT_SPEED_BW never fires: 0 of 8
# labelled shots were found. The audio track still holds the click of the kick. A click alone is not specific (every hold end has sharp
# noise nearby), so a hold end becomes a shot when it has BOTH a loud click AND the goal in view for a good part of the next 3 seconds.
# Derived on the user's 8 labelled shots + 340 hold ends in 18 videos (2026-11-08): click AND goal-in-view keeps ~13 candidates per
# 90 minutes and catches 4 of the 8 shots; goal-in-view alone gives 40 candidates for 6 shots. Only 8 positives: re-tune after more labels.
AUDIO_ENABLED = True              # needs `pip install av` and the `<video>_audio.npz` files (main.py creates them); without them this cue is skipped
AUDIO_KICK_PRE_S = 0.2            # the click is searched from this long before the hold ends ...
AUDIO_KICK_POST_S = 0.27          # ... to this long after
AUDIO_KICK_MIN_Z = 20.0           # robust z-score of the spectral flux that counts as a click (ordinary noise < 10, kicks 20-150)
AUDIO_SHOT_GOAL_FRAMES = 30       # goal detections needed in the AUDIO_SHOT_GOAL_WINDOW_FRAMES after the kick (30 = one second of goal in view)
AUDIO_SHOT_GOAL_WINDOW_FRAMES = 90
AUDIO_SHOT_MAX_ZONE_SIGHTINGS = 8 # veto: the ball seen this many times (or more) in the wearer's own zone in the 60 frames after the kick means he still has it (real shots: 0-4, false ones: 18-42)
AUDIO_SHOT_OUTCOME_DEFAULT = "on_target"   # outcome of an audio shot whose ball was not seen overlapping the goal: "on_target" or "off_target". Nothing in the data tells them apart (the ball is gone); 7 of the user's 8 labelled shots were on target, so "on_target" - a DECISION for the user
AUDIO_SCORE_FULL_Z = 100.0        # click strength (robust z) that earns the full "sound" score component of a shot
AUDIO_SHOT_PRIORITY = "over_pass" #"over_pass": a loud kick with the goal in view is a shot even if a receiver can be found; "after_pass": only when no pass was resolved

# ===========================================================================
# 4c. SHOT SELF-CHECK (stage B of the shot logic: SOFT evidence, never a hard gate)   -> src/shot_evidence.py
# ===========================================================================
# Stage A = the two rules above (ball flight towards the goal; loud click + goal in view). They stay exactly as they were and still decide on their own.
# Stage B looks at every kick candidate (each hold end, plus the last sighting of a big ball before a long gap) that stage A did not turn into a shot, and
# adds up five weak cues, none of them required: how big the goal looks (a shot is taken near it), how long it is in view around the kick, the click of
# the kick, a camera whip after the kick, and the ball being gone afterwards. The sum (0..1) can ADD a shot stage A missed or turn a pass into a shot, and it
# is stored on every shot as `shot_confidence`, which is one component of the shot's score (so a self-check shot ranks below a confirmed one and only
# reaches the highlight reel when there are few better ones).
# Why soft + hand-set (2026-11-14): on the 21 kicks the user's labels + my frame-by-frame review located (labels_refined_shots.csv), no single cue separates a
# kick from any other hold end (best AUCs: goal size 0.75, goal in view 0.76, click 0.69, whip 0.57, ball gone 0.58 on the calibration videos, and every
# one of them is weaker on the 4 held-out January videos) - so any hard threshold on one cue either misses most shots or fires on everything. The ramps below
# are physical scales (not fitted), the weights are round numbers, and the only dial that was set from data is SHOT_CHECK_MIN_SCORE, taken as a percentile of the
# scores of ALL candidates (not of the positives): "add at most about one possible shot every 10 minutes". Evidence score AUC: 0.82 calibration, 0.60 held-out.
SHOT_CHECK_ENABLED = True
SHOT_CHECK_MIN_SCORE = 0.78       # a candidate at or above this evidence score is a "possible shot" (98th..96th percentile of all candidates: ~0.1-0.2 per minute)
SHOT_CHECK_VANISH_GAP_FRAMES = 30 # a long gap without any ball sighting makes the last sighting a kick candidate ...
SHOT_CHECK_BIG_BALL_W = 0.03      # ... if the ball was at least this wide (share of the frame width) in the 30 frames before: it was near the camera
SHOT_CHECK_RAMPS = {              # cue -> (value that earns 0, value that earns the full 1); linear in between
    "goal_size": (0.08, 0.30),    # widest goal box in [kick-30, kick+90] as a share of the frame width (median 0.12 at ordinary hold ends, 0.26 at kicks)
    "goal_frames": (0, 40),       # frames with the goal in view in the 90 frames after the kick
    "click": (8, 60),             # robust z of the kick sound (+-0.25 s around the kick); this cue drops out of the sum when the video has no audio
    "whip": (60, 250),            # largest camera pan (|dx|+|dy|, scene px per frame) in the 25 frames after the kick
    "ball_gone": (20, 0),         # ball sightings in frames kick+5..kick+60 (0 sightings = gone = 1.0)
}
SHOT_CHECK_WEIGHTS = {"goal_size": 0.30, "goal_frames": 0.25, "click": 0.20, "whip": 0.10, "ball_gone": 0.15}
SHOT_CHECK_DEDUP_FRAMES = 90      # at most one possible shot per this many frames (the best one wins); a stage-A shot this close also blocks stage B

# ===========================================================================
# 5. EVENT SCORES (0-100) AND THE HIGHLIGHT SELECTION
# ===========================================================================
TOP_N_PER_EVENT = 3                  # at most this many of each event type go into the final video, best score first
SELECTION_SCOPE = "session"          # "session": best N across a whole session folder; "video": best N inside each video
DRIBBLE_SELECTION = "per_class"      # "per_class": best N of each of the 3 lengths (short, medium, long are three different events); "pooled": best N dribbles of any length

# Each event score is a weighted sum of components that are first scaled to 0..1. Weights need not sum to 1
# (they are normalised). See event_scoring.py for exactly what each component measures.
# Changed 2026-11-15 (user's decisions): dribble gets a "count" component (frames in which the ball was really detected); a pass that did not succeed is scored exactly like
# one that did (same weights, pressure included), so a hard pass that failed ranks like a hard pass that worked; shots weigh the kick sound and the ball speed most.
# The shot weights add up to 1.10 and the dribble ones to 1.05 - harmless, the weights are normalised.
SCORE_WEIGHTS = {
    "DRIBBLE": {"duration": 0.30, "count": 0.25, "visibility": 0.25, "pressure": 0.25},
    "PASS_SUCCESS": {"length": 0.30, "speed": 0.15, "pressure": 0.25, "team_confidence": 0.15, "visibility": 0.15},
    "PASS_INCOMPLETE": {"length": 0.30, "speed": 0.15, "pressure": 0.25, "team_confidence": 0.15, "visibility": 0.15},
    # the 0.10 slot is the goal-contact component of each kind: overlap with the goal (on target) / closeness to it (off target)
    "SHOT_ON_TARGET": {"sound": 0.40, "speed": 0.40, "check": 0.15, "overlap": 0.10, "visibility": 0.05},
    "SHOT_OFF_TARGET": {"sound": 0.40, "speed": 0.40, "check": 0.15, "closeness": 0.10, "visibility": 0.05},
}
# "count" of a dribble: SCORE_COUNT_FULL_FRAMES frames with a real ball detection earn the full 1.0, every frame fewer takes SCORE_COUNT_STEP off, never below SCORE_COUNT_MIN
# (50 frames: 1.00, 49: 0.98, ... 10: 0.20, 1: 0.02, 0: 0.01). 50 is the 95th percentile of the detected frames of the 297 holds measured 2026-11-15 (median 19, 75th 29, 90th 41).
SCORE_COUNT_FULL_FRAMES = 50
SCORE_COUNT_STEP = 0.02
SCORE_COUNT_MIN = 0.01
SCORE_LENGTH_FULL_PX = 400.0         # pass length (scene px at 960 wide) that earns the full "length" component
SCORE_SPEED_FULL_BW = 4.0            # ball speed (bw/frame) that earns the full "speed" component
SCORE_VISIBILITY_FULL = 0.5          # share of event frames with a real ball sighting that earns the full "visibility" component
SCORE_PRESSURE_FULL = 3              # opponents around the receiver/ball-carrier that earn the full "pressure" component
SCORE_DRIBBLE_FULL_FRAMES = 90       # hold length (3 s) that earns the full "duration" component
SCORE_CLOSENESS_FULL_GOAL_WIDTHS = 3.0  # an off-target shot that ends this many goal-widths away scores 0 closeness
