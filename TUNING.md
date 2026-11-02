# Tuning RehabQuest for your webcam

Every number below lives in **`.env`** (in this folder). Edit it, save, and restart the game. No code changes are needed.
Run with `python main.py --debug` (or press **D**) to see live values: FPS, the detected gesture, joint angles and
blendshape scores. That tells you which number to change.

## 1. First check (5 minutes)

| Check | How | Good result |
|---|---|---|
| Lighting | Face a window or lamp; avoid a bright window *behind* you | Calibration step 1 shows "Good light" |
| Distance | Sit ~1-1.5 m away, head + shoulders + elbows in view | "Good distance"; for Mirror Dancer your hands must stay in the picture when raised |
| Frame rate | `--debug` top line | FPS ≥ 25. If lower, set `DETECT_WIDTH=480` |
| Left/right hands | `--debug`, raise only your LEFT hand | Debug says "Left hand". If it says Right, set `HANDEDNESS_SWAP=1` |
| Face sides | Game 3, smile with only your LEFT mouth corner | The "Your left" bar rises. If "Your right" rises, set `FACE_SIDE_SWAP=0` |

## 2. Symptoms → constant to change

| Symptom | Constant (default) | Try |
|---|---|---|
| Hand cursor jittery | `SMOOTHING_ALPHA` (0.55) | 0.40 |
| Hand cursor feels laggy | `SMOOTHING_ALPHA` (0.55) | 0.70 |
| Buttons click too easily / too slowly | `HOVER_CLICK_SECONDS` (1.5) | 2.0 / 1.0 |
| "I can't see you" flickers on and off | `TRACKING_LOST_GRACE` (0.35) | 0.6 |
| Hands not detected in dim light | `MIN_DETECTION_CONFIDENCE` (0.5) | 0.4 |
| **Game 1** bubbles hard to pop | `G1_TOUCH_MARGIN` (28 px), `G1_BUBBLE_RADIUS` (62) | 40, 75 |
| Game 1 too long / short | `G1_PASS_POPS` (15), `G1_TIME_LIMIT` (90) | |
| **Game 2** open hand not recognised | `G2_OPEN_THRESHOLD` (0.72) | 0.60 (or recalibrate with **C**) |
| Game 2 fist not recognised (stiff fingers) | `G2_FIST_THRESHOLD` (0.30) | 0.40 |
| Game 2 holds too long for the patient | `G2_OPEN_HOLD` (2.0 s), `G2_FIST_HOLD` (1.0 s) | 1.5, 0.7 |
| **Game 3** smile never "strong enough" | `G3_SMILE_THRESHOLD` (0.45) | 0.35 |
| Game 3 eyebrows not detected | `G3_BROW_THRESHOLD` (0.35) | 0.25 |
| Game 3 eyes closed not detected (glasses!) | `G3_EYES_THRESHOLD` (0.50) | 0.40 |
| Game 3 kiss / mouth open not detected | `G3_PUCKER_THRESHOLD` (0.45), `G3_JAW_THRESHOLD` (0.40) | 0.35, 0.30 |
| Game 3 symmetry always too low | `G3_SYMMETRY_THRESHOLD` (0.55) | 0.45 (a real palsy patient may need 0.35) |
| Game 3 one-side smile not accepted | `G3_HALF_SMILE_DIFF` (0.20) | 0.15 |
| **Game 4** poses rarely match | `G4_TOLERANCE_SCALE` (1.0) | 1.3 (all angle tolerances 30% wider) |
| Game 4 matches too easily | `G4_POSE_ACCURACY` (0.75) | 0.85 |
| Game 4 not enough time per pose | `G4_POSE_WINDOW` (7.0 s) | 9.0 |
| Game 4 lean poses fail when seated | `G4_TOLERANCE_SCALE` | 1.3. Lean is measured hip-to-shoulder (falls back to shoulder tilt if hips are hidden) |
| **Game 5** pinches not recognised | `PINCH_THRESHOLD` (0.35) | 0.45 |
| Game 5 wrong finger detected in pinch | `PINCH_THRESHOLD` (0.35) | 0.28 |
| Game 5 "index only" fails (others not curled enough) | `FINGER_CURLED_MIN_DEG` (110°) | 95 |
| Game 5 straight finger counted as bent | `FINGER_EXTENDED_MAX_DEG` (80°) | 95 |
| Game 5 count 4 vs 5 confused (thumb) | `THUMB_EXTENDED_RATIO` (0.95) | 1.05 (thumb must spread further) |
| Game 5 too hard | `G5_MAX_MISTAKES` (3), `G5_STEP_TIME_END` (6 s), `G5_HOLD` (0.45) | 4, 8, 0.35 |
| Rest reminder too frequent | `REST_REMINDER_MINUTES` (10) | 15 |
| Pain-stop gesture triggers by accident | `PAIN_GESTURE_SECONDS` (1.5) | 2.5 |

## 3. Automatic help built in
* After **2 failed attempts in a row** a game's targets ease by `EASE_STEP` (10%) up to two levels, and the
  tutorial shows a tip. A 3-star pass removes one level again.
* **Calibration** (menu → Recalibrate, or **C**) re-measures reach and grip. Do it again if the patient, chair or
  camera position changes.

## 4. Reading the debug overlay
```
FPS  29.8   scene GameScene
hands 18ms
Right hand: pinch_middle  wrist=(0.62,0.71)  index=(0.58,0.44)
face: jawOpen=0.62  mouthSmileLeft=0.31 ...
pose: abdL=95 abdR=88 elbL=171 elbR=168 lean=-3  shrug=0.98
```
* `abdL/abdR`: shoulder lift in degrees (0 = arm down, 90 = T-pose, 180 = straight up).
* `elbL/elbR`: elbow straightness (180 = straight).
* `lean`: + = leaning to your right. `shrug` < 0.8 = shoulders raised.
