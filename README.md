# RehabQuest

**A webcam-controlled rehabilitation game suite: five games, five medical conditions, no mouse or keyboard needed.**

RehabQuest turns repetitive physiotherapy exercises into short, rewarding mini-games. A normal laptop webcam
tracks the patient's hands, face or upper body with Google MediaPipe. Each game asks for a real
rehabilitation exercise, measures it (range of motion, symmetry, reaction time, finger accuracy) and saves
the results so progress can be shown over weeks.

> **Disclaimer.** RehabQuest is a practice aid built for a college assignment. It does not diagnose and
> does not replace a physiotherapist, occupational therapist or doctor. Webcam measurements are approximate.
> Stop if anything hurts (cross your raised hands, or press **P**).

---

## Screenshots

| Journey map | Tutorial | Gameplay |
|---|---|---|
| *(add `docs/menu.png`)* | *(add `docs/tutorial.png`)* | *(add `docs/game.png`)* |

Tip: run `python main.py --demo` and press `Print Screen`, or use the Windows Snipping Tool.

---

## The five games

The patient starts at Game 1. **Each game is locked until the previous one is passed.**

| # | Game | Condition | Body part | Level | Mechanic | Pass criteria | Metrics recorded |
|---|------|-----------|-----------|-------|----------|---------------|------------------|
| 1 | **Bubble Garden** | Post-stroke arm weakness, reduced shoulder range | Shoulder & arm | EASY | Pop floating bubbles with your palm. They wait at a "reach line" that climbs towards your calibrated maximum reach and alternate left/right. Left, right or both hands. | Pop **15** bubbles in **90 s** | Reaction time, reach height (% of screen), left/right balance |
| 2 | **Lantern Lighter** | Rheumatoid arthritis, stiff fingers, weak grip, spasticity | Hand & fingers | EASY | **Open** the hand wide and hold 2 s to charge a lantern, then **close** a gentle fist for 1 s to light it. | Light **8** lanterns in **120 s** | Finger range of motion (open-fist ratio), hold time (endurance) |
| 3 | **Smile Sunshine** | Bell's palsy, facial nerve weakness, stroke facial droop | Face | EASY | A cartoon sun shows 6 facial exercises: big smile, raise eyebrows, close eyes, kiss/pucker, open wide, one-side smile. Copy and hold 2 s. | All **6** expressions held, with **symmetry ≥ 55%** | Left/right symmetry per expression, strength, time |
| 4 | **Mirror Dancer** | Parkinson's (LSVT-BIG "move big"), frozen shoulder, posture | Upper body | MEDIUM | A glowing ghost dancer appears on your own body. Match 12 poses (arms up, T, reach L/R, hands on head, lean L/R, shrug, elbows to ribs, single arm up) before the beat ring closes. | Match **10 of 12** poses at **≥ 75%** accuracy | Shoulder abduction & trunk lean (range of motion), time to match, posture score |
| 5 | **Spell Caster** | Advanced stroke recovery, hand surgery, carpal tunnel | Fingers, both hands | HARD | Memorise a spell of finger "runes" (thumb-to-finger taps, single finger raises, counting 1-5, a final two-hand gesture), then cast it in order. Spells grow from 2 to 6 runes. | **5** spells with **≤ 3** mistakes | Per-finger success rate & speed (weakest finger), sequence memory |

## Progression

```
 First run: 30-second calibration (camera check, weaker side, reach, grip)
                                   |
                                   v
 [1 Bubble Garden] --> [2 Lantern Lighter] --> [3 Smile Sunshine] --> [4 Mirror Dancer] --> [5 Spell Caster]
      EASY                  EASY                    EASY                  MEDIUM               HARD
        \___ pass to unlock ___/      \___ pass ___/        \___ pass ___/       \___ pass ___/

 Every game:  Tutorial (animated demo + 3 instructions)  ->  "show your hand to start"  ->  3-2-1-GO
              -> play  ->  results (stars, accuracy, mini chart)  ->  next game unlocked / retry with hint
 Fail twice in a row -> targets ease by 10% (max 2 levels) + tip.   3 stars -> "Ready for the next challenge?"
```

Stars: 1 = passed, 2 = passed quickly or nearly perfectly, 3 = fast and accurate (each game defines its own).

---

## How it works (technical)

### Architecture
```
rehabquest/
  main.py               App + state machine: Loading -> Calibration -> Menu -> Tutorial -> Game -> Result
  .env                  every tunable constant (thresholds, timings, camera) - see TUNING.md
  core/
    config.py           loads .env with typed defaults
    camera.py           threaded webcam capture, mirror + 16:9 crop; DemoCamera
    trackers.py         MediaPipe Tasks (Hand/Face/PoseLandmarker), model auto-download, smoothing, AsyncTracker
    gestures.py         finger angles, openness, pinch, gesture library            (pure functions, unit-tested)
    face_metrics.py     expression strength + left/right symmetry                   (pure functions, unit-tested)
    pose_metrics.py     joint angles, pose library with tolerances, posture        (pure functions, unit-tested)
    synth.py            synthetic hand/face/body generator: demo mode, tutorial animations, tests
    ui.py               theme, TrueType text cache, panels, hover buttons, icons, glowing skeletons
    effects.py          particles, confetti, shockwaves, floating score text, screen shake, easing
    audio.py            sound effects + ambient music generated with NumPy (no audio files)
    progress.py         progress.json (unlocks, stars, ease level) + sessions.csv
    report.py           matplotlib charts + HTML report with therapist-style comments
    calibration.py      first-run calibration screen
    scenes.py           menu journey map, tutorial, results, rest, pain stop, report screens
  games/
    base_game.py        BaseGame: intro(), update(frame, landmarks, dt), draw(frame), result(),
                        timer that pauses when tracking is lost, score, combo, HUD, stars, easing
    game1_bubble_garden.py ... game5_spell_caster.py
  tests/                pytest: gestures, face, pose, progression, report, every game, full app
  models/  data/  reports/  logs/      created at runtime
```

### Landmarks
MediaPipe returns 21 hand landmarks (also as **world** coordinates in metres), 478 face landmarks with 52
**blendshape** scores (0-1), and 33 body landmarks with visibility. The camera image is mirrored first, so
the patient's left is on the left of the screen, like a mirror.

### The maths
* **Hand size** `s = |wrist - middle-finger knuckle|`. Every hand distance is divided by `s`, so results are the
  same at any distance from the camera.
* **Finger bend**: the sum of the three joint angles along a finger,
  `bend = ∠(wrist→MCP, MCP→PIP) + ∠(MCP→PIP, PIP→DIP) + ∠(PIP→DIP, DIP→TIP)`.
  Straight < 80°, curled > 110°. Angles are computed in 3-D world space, so they work **in any hand
  orientation**. A test rotates synthetic hands randomly in 3-D to prove it.
* **Openness** (Game 2): `mean(|wrist - fingertip|) / s` for the four fingers (≈1.9 open, ≈1.0 fist), rescaled
  to the patient's own calibrated open/fist range: `openness = (r - r_fist) / (r_open - r_fist)`.
* **Pinch** (Game 5): `|thumb tip - finger tip| / s < 0.35`, and the finger must be clearly closer than the
  runner-up, so thumb-to-middle is never mistaken for thumb-to-index.
* **Facial symmetry** (Game 3): `symmetry = min(L, R) / max(L, R)` from paired blendshapes
  (`mouthSmileLeft/Right`, `browOuterUpLeft/Right`, `eyeBlinkLeft/Right`). For pucker and mouth-opening, which
  have no left/right blendshape, each mouth corner's distance from the face midline (nose bridge to chin) is
  compared instead, penalised by any height difference between the corners.
* **Joint angles** (Game 4): shoulder **abduction** = angle between the torso-down direction (perpendicular to
  the shoulder line) and the upper arm; **elbow** angle at the elbow (foreshortened forearms count as bent);
  **trunk lean** from hips to shoulders; **shrug** = neck length / relaxed baseline. Each pose is a list of
  criteria (`min` / `max` / `near` target ± tolerance). Accuracy is their weighted score, and the worst
  criterion becomes the on-screen hint ("Lift your LEFT arm higher"). Left and right are decided by screen
  position, not by model labels.
* **Smoothing**: an adaptive exponential moving average. Still hands are smoothed heavily (no jitter), fast
  movements lightly (low lag).
* **Performance**: detection runs on a background thread, text is pre-rendered and cached, and glow sprites are
  cached. Drawing takes about 6-13 ms per frame, so the game runs at 30 FPS on an ordinary laptop.

### Personalisation
The 30-second calibration measures the weaker side, highest comfortable reach (left and right), open and fist
hand ratios, and relaxed neck length. Bubble heights, lantern thresholds and shrug detection all scale to
these, so targets fit the patient rather than one size for everyone.

---

## Installation & running

Requires **Python 3.10 or 3.11** (MediaPipe does not support every newer version) and a webcam.
The three MediaPipe models (~17 MB) download automatically on first run into `models/`.

### Windows
```bat
double-click run.bat
```
or manually:
```bat
py -3.10 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python main.py
```

### macOS / Linux
```bash
./run.sh
# or: python3.10 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/python main.py
```
On macOS, allow camera access for your terminal in *System Settings → Privacy & Security → Camera*.

### Command-line options
| Command | What it does |
|---|---|
| `python main.py` | Normal play with the webcam |
| `python main.py --demo` | **No webcam needed.** A virtual patient plays each game; in menus the mouse moves a virtual hand |
| `python main.py --demo --autoplay` | Hands-free presentation: the virtual patient plays the whole journey, then opens the report |
| `python main.py --debug` | Shows FPS, tracker timings and raw landmark values (also toggle with **D**) |
| `python main.py --selftest` | Headless run through calibration + all 5 games; prints a results table |
| `python main.py --reset` | Reset game progress (keeps calibration) |
| `python main.py --camera 1` | Use a different camera |
| `python -m pytest` | Run the 120+ unit / smoke tests |

Demo mode keeps its data in `data_demo/`, so real patient data in `data/` is never mixed with simulated data.

## Controls

Everything works by **hand**: hold your hand (index fingertip) over a button for 1.5 s and a ring fills to click.

| Key | Action |
|---|---|
| `1`-`5` | Choose a game (menu) |
| `SPACE` / `Enter` | Start / continue |
| `ESC` | Back to menu / quit from menu |
| `M` | Mute / unmute |
| `F` | Fullscreen on/off (window is also resizable) |
| `P` | **Pain stop**: stop immediately (also: cross both raised hands for 1.5 s) |
| `N` | Teacher-demo shortcut: mark the current game as passed and unlock the next |
| `R` / `C` | Progress report / recalibrate (menu) |
| `L` / `R` / `B` | Left / right / both hands (Bubble Garden tutorial) |
| `D` | Debug overlay |

## Data and reports
* `data/progress.json`: unlocked games, stars, best scores, ease level, calibration.
* `data/sessions.csv`: one row per game: date, game, score, accuracy, reaction time, range of motion, duration,
  pass/fail, stars, plus game-specific details (left/right pops, per-expression symmetry, per-finger success...).
* **Progress Report** (menu): saves `reports/report_<date>.png` and `.html` and opens the HTML. It contains score
  trends, left vs right reach, a facial symmetry trend, finger success rates (weakest finger), shoulder range and
  finger range, plus plain-language comments such as *"Your shoulder reach improved 12% this week."*

## Safety & robustness
* Visible disclaimer on every menu screen; **pain stop** gesture/key; **rest reminder** after 10 minutes of play.
* No webcam or camera busy → friendly screen with *retry* or *demo mode*. Camera unplugged mid-game → reconnect message.
* Tracking lost → "I can't see your hand, please move into view" and **the timer pauses**.
* Several faces or hands → the closest (largest) is used.
* Any unexpected error is logged to `logs/error.log` and the app returns to the menu instead of crashing.

## Testing
No webcam exists in a test environment, so `core/synth.py` generates anatomically plausible synthetic
landmarks. The test suite checks:
* every gesture is recognised, for both hands, under **random 3-D rotations** and at different hand sizes;
* each pose matches itself and a relaxed body matches none; left/right does not depend on model labels;
* the symmetry formula, lopsided smiles are caught, easing works;
* unlock rules, ease after two fails, corrupt save files recover, report generation;
* every game is **played to completion by the virtual patient** and passes; the timer pauses with nobody in view;
* the full app runs headless from calibration through all five games (`--selftest`).

## Known limitations
* A 2-D webcam cannot see depth well: movements straight towards the camera are under-measured.
* Lighting, skin tone, sleeves and busy backgrounds affect MediaPipe accuracy. Use even, front-facing light.
* Facial blendshapes vary between people. Thresholds may need tuning per patient (see `TUNING.md`).
* Angles are *approximate* and not a clinical goniometer. Use the trends, not single numbers.
* Hand labels (left/right) from MediaPipe can occasionally swap; `HANDEDNESS_SWAP` in `.env` fixes a consistent swap.

## Ethics
Data never leaves the computer: no accounts, no cloud, no video is recorded (only numbers in a CSV). The games
encourage rather than punish (escaped bubbles cost nothing, hints after failure, automatic easing), because
discouragement is a real barrier in rehabilitation. A clinician should choose which games suit a patient.

## Future work
* Therapist dashboard: assign games and custom thresholds per patient, export to PDF.
* Depth (stereo or RGB-D camera) for true 3-D joint angles.
* Lower-limb games (sit-to-stand, marching, balance) using the full pose model.
* Voice instructions and multiple languages for low-vision patients.
* A clinical validation study comparing webcam angles with a goniometer.
