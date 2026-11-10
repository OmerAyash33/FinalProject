# Held-out audit, 2026-11-08 (Claude's reading of contact sheets, NOT the user's labels)

Videos: January 2026-01-30, no. 2, 4, 6, 8 (36 min, 4K, another day than every video the parameters were tuned on). Parameters: pass 0.25 / 2.5 / 4.0, audio shot cue with both vetoes.
Stage 2 found 72 events in them: 24 passes, 6 shots, 42 dribbles. Contact sheets: frames f-10 .. f+60 every 10 frames.

## Shots (all 6 looked at) - 0 clear shots
| video | frame | what the sheets show |
|---|---|---|
| 2 | 5192 | ball at the feet, then gone, camera turns to the field with a goal at the edge. Long kick, could be a shot (unclear) |
| 2 | 9546 | ball at the feet, gone, a small ball seen far on the field 70 frames later. Long kick / pass (unclear, no goal clearly in view) |
| 4 | 4917 | opponent in a red shirt tackles / crowds the wearer, the camera ends up covered. Not a shot |
| 4 | 5841 | ball rolls slowly away over the pitch towards players, seen for 50 frames. Pass, not a shot |
| 6 | 581 | ball kicked forward, seen mid-field 50 frames later. Pass / long ball, not a shot |
| 8 | 11299 | ball played towards a player, a neon-vest player covers the view. Pass, not a shot |

## Passes (8 of 24, random sample seed 7)
real pass to a player: 4 of 8 (v4 1459, v4 6724, v6 6604 probable, v2 399 unclear); false: 4 of 8 (v2 1155 ball kicked away with nobody near, v4 13863 resting ball at the rest area, v6 14459 an opponent wins the ball (tackle), v8 8347 a kick with no receiver in view).
The completed/incomplete class could not be checked: nobody told us which January jerseys are the user's.

## Reading
* The audio+goal shot rule that reached 50% precision on the calibration videos finds no clear shot on a new day: 'goal in view' is true for 36-64 of 90 frames on these small fields almost every time the ball leaves the feet, and a kick click is not specific.
* Pass precision on the new day is about 50% in this sample, in line with the calibration lower bound.
* The held-out numbers in benchmark_history.csv stay empty until the user labels these four videos (tool combo box: 2026-01-30, videos 000002, 000004, 000006, 000008).
