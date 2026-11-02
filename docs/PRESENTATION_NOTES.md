# RehabQuest: presentation notes

## Before you present (checklist)
- [ ] `run.bat` (or `python main.py`) works and the models are already downloaded (`models/` has 3 files).
- [ ] Good light on your face, laptop at eye level, chair ~1.2 m from the camera.
- [ ] Sound on. A second terminal is ready with `python main.py --demo --autoplay` as a backup.
- [ ] Optional: run `python main.py --reset` so the journey starts at Game 1 (calibration is kept).
- [ ] Optional: play each game once beforehand so the Progress Report has data.

## 3-minute demo script

| Time | Do | Say |
|---|---|---|
| 0:00 | Show the **journey map** | "RehabQuest is five webcam games for five rehabilitation needs: stroke, arthritis, facial palsy, Parkinson's and fine-motor recovery. No mouse or keyboard: I click by holding my hand over a button." *Hover over Game 1 until the ring fills.* |
| 0:20 | **Tutorial** screen | "Every game teaches itself: the condition, an animated demo, three instructions in big text, and it starts only when it can see me." |
| 0:30 | Play **Bubble Garden** ~30 s | "Bubbles wait at a reach line that rises towards the maximum reach measured in my calibration, so it trains shoulder range. It records reaction time and left/right balance." |
| 0:60 | Press **N** to skip to unlock, open **Lantern Lighter** briefly | "Open wide to charge, gentle fist to light. Openness is fingertip distance divided by hand size, so it works at any distance." |
| 1:15 | **Smile Sunshine** ~20 s | "For Bell's palsy. The key number is this symmetry bar: min of left and right over max, from MediaPipe face blendshapes." |
| 1:35 | **Mirror Dancer** ~25 s | "Inspired by LSVT-BIG for Parkinson's. The ghost dancer sits on my own shoulders. Accuracy comes from joint angles like shoulder abduction and trunk lean, with a hint naming my worst joint." |
| 2:00 | **Spell Caster** ~20 s | "The hard level: memorise a finger sequence, then cast it. Finger states use joint angles, so it works with the hand rotated. It finds my weakest finger." |
| 2:20 | Cross both raised hands | "Safety: crossing my hands is a pain stop. There's also a rest reminder after 10 minutes and a disclaimer on every screen." |
| 2:30 | Menu → **Progress Report** | "Every session is saved to a CSV. The report shows trends, left vs right, symmetry, weakest finger, and writes comments like a therapist would." |
| 2:50 | Show terminal: `python -m pytest` | "I can't unit-test with a real webcam, so I built a synthetic body generator. Over 120 tests, including a virtual patient that plays all five games. Thank you!" |

**If the webcam fails:** run `python main.py --demo --autoplay` and narrate over it. The virtual patient plays everything.

## 10 likely questions and answers

1. **How does the camera know my fingers' positions?**
   Google MediaPipe's Hand, Face and Pose Landmarker models (the current Tasks API). They return 21 points per hand,
   478 per face (plus 52 expression scores) and 33 per body, every frame, on the CPU.

2. **Why does distance from the camera not matter?**
   All hand measurements are divided by the hand's own size (wrist to middle knuckle), and angles do not depend on size.
   A test scales synthetic hands from 0.7× to 1.4× and gets identical results.

3. **What if the patient turns their hand?**
   Finger states use the bend angles between finger segments in 3-D world coordinates, not "is the tip above the knuckle".
   A test rotates hands randomly in 3-D and still recognises every gesture.

4. **How is facial symmetry calculated?**
   For paired muscles: `min(left, right) / max(left, right)`, so 1.0 is perfectly even. For pucker and mouth-opening
   I compare each mouth corner's distance from the face midline (nose bridge to chin).

5. **Is it personalised?**
   Yes. A 30-second calibration measures the weaker side, maximum reach, open/fist hand range and relaxed neck length.
   Bubble heights, lantern thresholds and shrug detection scale to that. After two fails in a row, targets ease by 10%.

6. **How did you test without a webcam?**
   `core/synth.py` builds anatomical hands, faces and bodies from parameters (finger curls, joint angles, blendshapes).
   The same generator powers demo mode, the tutorial animations, the ghost dancer, and the tests: gesture recognition,
   pose matching, progression rules, and a virtual patient that plays all five games end to end.

7. **Is it medically accurate?**
   It is a motivation and practice aid, not a diagnostic tool. A 2-D webcam under-measures movement towards the camera.
   That's why the report focuses on *trends* over time and why there's a clear disclaimer. Validation against a
   goniometer is listed as future work.

8. **What about privacy?**
   No video is recorded or uploaded. Only numbers (score, angles, times) are saved locally in `data/sessions.csv`.

9. **How does it stay smooth at 30 FPS?**
   Tracking runs on a background thread. Text is pre-rendered once and cached, and glow effects use cached sprites. A
   frame takes about 6-13 ms to draw. Landmarks use an adaptive moving average: smooth when still, responsive when moving.

10. **How would you extend it?**
    Lower-limb games (sit-to-stand, marching), a therapist dashboard to assign games and thresholds, voice instructions,
    and a validation study. New games only need to subclass `BaseGame` and add one line to `games/registry.py`.
