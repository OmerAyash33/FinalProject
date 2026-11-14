# Progress report 2026-11-14 13:17

Detector v2; 227 labels of the user in 35 videos. Same labels for every run below, so the rows are comparable.

## 1. Progress on the user's labels

### Calibration videos (31 videos, 174 min)

| family | run | labels found | recall | class acc | events | precision (lower bound) | events/min |
|---|---|---|---|---|---|---|---|
| passes | START (old settings) | 13 of 70 | 0.19 (0.11-0.29) | 0.69 | 51 | 0.26 | 0.29 |
| passes | BEFORE (2026-11-08 settings) | 25 of 70 | 0.36 (0.26-0.47) | 0.72 | 106 | 0.26 | 0.61 |
| passes | NOW (+ shot self-check) | 25 of 70 | 0.36 (0.26-0.47) | 0.72 | 106 | 0.26 | 0.61 |
| shots | START (old settings) | 0 of 17 | 0.00 (0.00-0.18) | - | 0 | - | 0.00 |
| shots | BEFORE (2026-11-08 settings) | 5 of 17 | 0.29 (0.13-0.53) | 1.00 | 20 | 0.25 | 0.12 |
| shots | NOW (+ shot self-check) | 7 of 17 | 0.41 (0.22-0.64) | 1.00 | 33 | 0.24 | 0.19 |
| dribbles | START (old settings) | 66 of 86 | 0.77 (0.67-0.84) | 0.39 | 266 | 0.34 | 1.53 |
| dribbles | BEFORE (2026-11-08 settings) | 66 of 86 | 0.77 (0.67-0.84) | 0.39 | 266 | 0.34 | 1.53 |
| dribbles | NOW (+ shot self-check) | 66 of 86 | 0.77 (0.67-0.84) | 0.39 | 266 | 0.34 | 1.53 |

### HELD-OUT January videos 2/4/6/8 (36 min, never tuned on)

| family | run | labels found | recall | class acc | events | precision (lower bound) | events/min |
|---|---|---|---|---|---|---|---|
| passes | START (old settings) | 4 of 20 | 0.20 (0.08-0.42) | 0.50 | 11 | 0.36 | 0.31 |
| passes | BEFORE (2026-11-08 settings) | 8 of 20 | 0.40 (0.22-0.61) | 0.50 | 24 | 0.33 | 0.67 |
| passes | NOW (+ shot self-check) | 8 of 20 | 0.40 (0.22-0.61) | 0.50 | 24 | 0.33 | 0.67 |
| shots | START (old settings) | 0 of 5 | 0.00 (0.00-0.43) | - | 0 | - | 0.00 |
| shots | BEFORE (2026-11-08 settings) | 0 of 5 | 0.00 (0.00-0.43) | - | 6 | 0.00 | 0.17 |
| shots | NOW (+ shot self-check) | 1 of 5 | 0.20 (0.04-0.62) | 1.00 | 12 | 0.08 | 0.33 |
| dribbles | START (old settings) | 16 of 22 | 0.73 (0.52-0.87) | 0.44 | 42 | 0.36 | 1.17 |
| dribbles | BEFORE (2026-11-08 settings) | 16 of 22 | 0.73 (0.52-0.87) | 0.44 | 42 | 0.36 | 1.17 |
| dribbles | NOW (+ shot self-check) | 16 of 22 | 0.73 (0.52-0.87) | 0.44 | 42 | 0.36 | 1.17 |

### Did we progress? (recall, same labels)

| family | split | START | BEFORE | NOW | verdict |
|---|---|---|---|---|---|
| passes | calib | 13/70 | 25/70 | 25/70 | clear progress (p = 0.018) |
| passes | heldout | 4/20 | 8/20 | 8/20 | probably progress, not proven (p = 0.15) |
| shots | calib | 0/17 | 5/17 | 7/17 | clear progress (p = 0.004) |
| shots | heldout | 0/5 | 0/5 | 1/5 | up, but within noise (p = 0.50) |
| dribbles | calib | 66/86 | 66/86 | 66/86 | no change |
| dribbles | heldout | 16/22 | 16/22 | 16/22 | no change |

Reading it: recall = share of the user's labels that an event of the same family matches (event start - 15 <= label <= event end + 90 frames); precision is only a lower bound (the user may have missed some real events); the interval after the held-out recall is the Wilson 95% interval: with 5 shots or 20 passes it is wide, so a difference of one or two events is NOT proof of progress.

## 2. What would reach the highlight reel (top-3 shots per kind and session)

| split | self-check OFF: shots in reel / real | self-check ON: shots in reel / real |
|---|---|---|
| calib | 7 / 2 | 11 / 3 |
| heldout | 3 / 0 | 3 / 1 |

"Real" = a label of the user within the matching window. These are single-digit counts.

## 3. Videos nobody has seen (no labels: counts and rates only)

- **day**: 5 videos, 39 min, 84 events -> per minute {'DRIB': 0.95, 'PASS': 0.9, 'SHOT': 0.31}; shots by origin {'audio+goal': 8, 'self_check': 4}, median shot confidence 0.75
- **night**: 5 videos, 25 min, 49 events -> per minute {'DRIB': 1.2, 'PASS': 0.72, 'SHOT': 0.04}; shots by origin {'audio+goal': 1}, median shot confidence 0.65

For comparison, the labelled sessions with the current rules (events per minute): day: {'DRIB': np.float64(1.12), 'PASS': np.float64(0.57), 'SHOT': np.float64(0.3)} (95 min); night: {'PASS': np.float64(0.66), 'DRIB': np.float64(1.76), 'SHOT': np.float64(0.15)} (115 min)

### Manual check of the shots found in the unseen videos

Check of ALL 13 shots the pipeline found in the 10 unseen videos (2026-11-14). CORRECTED the same day after the user's remark.

What happened: my first check looked only at contact sheets of 11 frames around each event (tools/frame_sheet.py) and judged by the pictures, without the sound.
The user then said that all 13 are real kicks, and asked whether the click loudness was checked. It had not been. A picture cannot show a kick that happens next to the
wearer or just out of frame, and cannot hear it. The ground truth for this table is therefore the USER'S statement (all 13 are real kicks); my picture verdicts are kept only to show how far
they were off. "click z" = robust z-score of the audio spectral flux at the kick (AudioTrack.kick_z; ordinary noise < 10, kicks 20-150); "rank" = the share of all clicks >= 20 in that video that are quieter than this one.

| # | set | video | frame | origin | confidence | click z | rank in video | my picture verdict (superseded) |
|---|---|---|---|---|---|---|---|---|
| 1 | day | 20251031091117_000006 | 13805 | audio+goal | 0.55 | 20.3 | 1% | no |
| 2 | day | 20251216084606_000006 | 785 | audio+goal | 0.77 | 102.5 | 97% | unclear |
| 3 | day | 20251216084606_000006 | 1863 | audio+goal | 0.64 | 63.1 | 92% | no |
| 4 | day | 20251216084606_000006 | 5050 | self-check | 0.79 | 23.2 | 22% | unclear |
| 5 | day | 20251216084606_000006 | 8053 | audio+goal | 0.81 | 34.7 | 65% | unclear |
| 6 | day | 20260220084805_000005 | 127 | audio+goal | 0.65 | 27.8 | 41% | no (goalkeeper's clearance - the user: counts, no special case) |
| 7 | day | 20260220084805_000005 | 10689 | audio+goal | 0.73 | 23.8 | 26% | unclear |
| 8 | day | 20260220091505_000008 | 9307 | audio+goal | 0.69 | 72.3 | 92% | likely real |
| 9 | day | 20260220091505_000008 | 10551 | self-check | 0.83 | 91.4 | 97% | unclear / no |
| 10 | day | 20260220091505_000008 | 10780 | self-check | 0.93 | 61.0 | 89% | likely real |
| 11 | day | 20260220091505_000008 | 15674 | self-check | 0.83 | 36.6 | 62% | unclear |
| 12 | day | 20260220091505_000008 | 16034 | audio+goal | 0.60 | 82.7 | 96% | no |
| 13 | night | 20260505205610_000030 | 1078 | audio+goal | 0.65 | 67.0 | 91% | plausible |

Result: 13 of 13 are real kicks (the user). All 13 have a click >= 20 at the kick, 7 of them 61-103 (among the loudest 10% of the clicks in their video), 4 are weak (20-28).
My picture verdicts were wrong at least for #12 (click 83, "nothing happens") and #9 (click 91): the old "15-25% real" figure for this set is WRONG and is withdrawn.

What the sound does and does not tell (measured on the labelled videos, 911 candidate moments = hold ends and "ball vanished" points, of which 20 are the user's shots):
- It confirms that a kick happened, it does not say that the kick is a shot: 32% of all candidates have a click >= 20 (passes, touches, clearances are kicks too). Alone, "click >= 20" would
  pick 294 candidates and catch 10 of the 20 user shots; AUC click vs the rest 0.72 on the calibration videos, 0.54 on the 4 held-out ones.
- For half of the user's shots there is NO click at the candidate moment (10 of 20 with z < 20 in the window -0.2/+0.27 s). With a window of -0.5/+0.5 s it is 16 of 20, but then 52% of the
  other candidates have one as well (AUC unchanged, 0.71): the candidate moment is off by 0.3-0.5 s for about a third of the shots, and widening the window does not separate better.
- Labelled videos, shots the pipeline found but the user did not label (36): 11 overlap a pass or dribble label of the user (a real kick, classified differently), 25 have no label at all and 22 of those
  have a click >= 20. So the old "precision about 20%" of the shot events was a lower bound dominated by kicks that were simply not labelled.
- Still open: goal kicks and clearances are now counted as shots by the user's decision; shots by OTHER players next to the wearer (#10) are found only because of the click + goal.

## 4. What to expect on the next unseen session (simulation from the held-out numbers; intervals, not promises)

- passes: recall about 40% (plausible range 22%-61%), about 0.67 events per minute, at least 33% of them real
- shots: recall about 20% (plausible range 4%-62%), about 0.33 events per minute, at least 8% of them real
- dribbles: recall about 73% (plausible range 52%-87%), about 1.17 events per minute, at least 36% of them real
- shots, the 13 shots found in the unseen videos: the user says all 13 are real kicks (my first picture-only check said 15-25%, withdrawn: it did not use the sound); the open question there is recall (how many kicks are missed), which no one can measure without labels
- the final blind session: expect these numbers or slightly worse; the benchmark's 'precision' of shots is only a lower bound, 25 of 36 unmatched shot events in the labelled videos have no label at all and 22 of them a loud click

