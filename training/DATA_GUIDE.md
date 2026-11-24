# Producing more training data and media

Our experiments so far point the same way. A larger input, a finer P2 grid and the big teacher model did not find more balls. The model does well on the match it trained on and much worse on a new one (small-ball recall 69 % on val, 37 % on the held-out match), and 60 % of missed balls produce no candidate at all. **New, varied footage is the main lever.** This guide covers filming it, turning it into labels, and handling the people in it.

## 1. Before filming: privacy, consent and de-identification

*Not legal advice. Check the rules where you film, and the venue's or league's own policy.*

**Footage of people is personal data, and blurring does not make it anonymous.** Players stay identifiable by kit, shirt number, build, gait and context (your club, that pitch, that date), and the raw video exists before any blur. Street-imagery services such as Google Street View don't rely on blurring alone either. They combine:
- automatic blurring tuned to miss as few faces and plates as possible
- an irreversible blur
- a legal basis and public notice
- a way for anyone to ask for more blurring or removal

This pipeline follows the same model (section 6). The legal part depends on where you are:

| Where | Baseline |
|---|---|
| **EU, UK (GDPR)** | You need a legal basis. Consent is one; a club filming its own matches may rely on *legitimate interest* instead, which still needs clear notice to players, an easy opt-out and a documented balancing test. Anything you contribute to the author's dataset falls under his GDPR obligations (France). |
| **US** | Filming in public places is generally lawful without consent; there's no general federal privacy law. But venues and leagues often have filming policies, using someone's image commercially can raise right-of-publicity claims, and a few states have biometric laws (the ball model doesn't do face recognition). |
| **Anywhere: minors** | Children get extra protection almost everywhere, and youth leagues and schools usually have their own rules. **Don't film youth matches for training without parents' or guardians' agreement**, whatever the blurring. |

**The practical baseline for your own club's matches**, which covers most situations:
1. **Tell everyone filmed** what is filmed and why: a ball-tracking model, kept locally, faces blurred. A team-chat message is enough.
2. **Offer an opt-out**, and honour it: don't film that match, or delete that person's frames.
3. **De-identify at extraction** (on by default, section 6). Unblurred frames are never written to disk.
4. **Keep the raw video off shared and cloud storage**, and delete it once frames are extracted and checked, unless you need it again.
5. **Removal on request:** one folder per match makes it easy to delete a person's frames, or a whole match.

For anything beyond your own club's matches (other teams, publishing, minors), get the venue's or league's agreement, or proper advice.

**Publishing** (a dataset contribution, demo clips, social media) is a stronger step than training locally. It needs de-identified frames *and* the players' agreement to publication. For a CC BY 4.0 dataset contribution you also need the rights to the footage.

## 2. What to film: variety beats volume

One more match like the training match adds little; matches *unlike* it help. Aim for coverage:

| Dimension | Aim for |
|---|---|
| Venue | Several pitches, not just your home ground. Each new venue is worth more than more minutes at a known one. |
| Light | Daylight, low sun and shadows, dusk, floodlights. Night is where the author's footage is strongest. |
| Weather | Dry, rain, wet grass glare if you get it. |
| Ball | White, high-visibility yellow or orange (winter balls), and different patterns. |
| Pitch and format | Grass and artificial; 5-, 7- and 11-a-side. Pitch size changes how small the ball looks. |
| Camera | Touchline at chest height, raised (stand, fence, bleacher), halfway line and behind a goal. Ultra-wide (×0.5) and main lens. |
| Motion | The app on the gimbal (the real use case) and some fixed-camera clips. |
| Scene clutter | Spare balls, cones, bags, other matches in the background. These are the false positives that remain. |

Rough targets for a first round: **3 to 5 matches at 2 or more venues, including one night match.** Keep one of them as a **reserved test match**: never used for training, labelled fully by hand. Pick one that differs from the training data, ideally a new venue.

Extras that are cheap and useful:
- **Warm-ups and drills** (many balls, close and far)
- **Empty-pitch or no-ball clips** (true negatives)
- **Clips with spare balls and cones lying around**

## 3. Filming with the app

- **Setup:** the iPhone on the DockKit gimbal, with the box2boxcam app recording. The app saves the video plus `events.jsonl` (detections, decisions, motor commands) and `highlights.json`. Keep the logs with the video: they show what the model did live.
- **Video settings:** 1080p at 30 fps, as the app records. That's enough: the model sees 960 px. Do not use 4K for training footage, as it only costs storage.
- **Lens:** the ultra-wide (×0.5) on small pitches, the main lens on large ones, as the app suggests. Note which you used.
- **Power:** a full match used about 65 % of the phone's battery in the author's test. Bring a power bank or start full.
- **Storage:** about 3 to 5 GB per hour at 1080p30.
- **During the match:** don't touch the phone. Odd camera behaviour is useful data for the app too.
- **After the match:** export the video and the logs from the app's recordings library ("Vidéo + logs").

## 4. Getting the footage to the training machine

Without cloud services:
1. Connect the iPhone to the Windows PC by USB and import the videos (Windows Photos app, or File Explorer → Apple iPhone → DCIM). Or AirDrop them to the Mac first and copy them over.
2. In WSL the Windows drive is at `/mnt/c/`. Copy each match into its own folder, for example:

   ```sh
   mkdir -p ~/videos/club_20261130
   cp "/mnt/c/Users/<you>/Pictures/<import folder>/IMG_1234.MOV" ~/videos/club_20261130/
   ```

3. Name every match `<place>_<YYYYMMDD>`, one id per match, used everywhere afterwards.

## 5. A match sheet for every match

Keep a line per match in `~/videos/matches.csv`. It's needed later to explain results ("night matches got better, yellow ball did not"):

```
match,date,venue,format,light,weather,ball,lens,camera_position,camera_motion,role,consent,notes
club_20261130,2026-11-30,Home ground,7v7,floodlights,dry,white,x0.5,touchline chest height,gimbal app,train,team chat 2026-11-28,
park_20261207,2026-12-07,City park,11v11,daylight low sun,wet,yellow,x1,raised fence,gimbal app,reserved,signed forms,new venue
```

## 6. From video to labels

The pipeline is in `teacher/` (see `teacher/README.md`). Per match:

```sh
cd ~/workspace/football-vision/training
# training match: 1 frame per second, 5 % of empty frames to review, 10 % of auto-accepted frames audited
.venv/bin/python -I teacher/pseudo_label.py --video ~/videos/club_20261130/IMG_1234.MOV --match club_20261130 \
    --role train --teacher eval/models/base_s42.pt eval/models/base_s7.pt --flip \
    --thresholds teacher/thresholds.json --stride 30 --review-empty 0.05 --audit-auto 0.1
# reserved test match: every sampled frame goes to human review
.venv/bin/python -I teacher/pseudo_label.py --video ~/videos/park_20261207/IMG_2001.MOV --match park_20261207 \
    --role reserved --teacher eval/models/base_s42.pt eval/models/base_s7.pt --flip \
    --thresholds teacher/thresholds.json --stride 60
```

**De-identification (on by default).** Every sampled frame is de-identified *before* it is written to disk (`teacher/head_blur.py`):
- **Detection:** a pose model and a person and vehicle detector run on the whole frame and on 4 overlapping tiles, so distant players are found too.
- **Heads:** every person's head is pixelated then blurred, faces and back-turned heads alike.
- **Vehicles:** the lower half of every vehicle is blurred, where number plates are.

The teacher, the Label Studio reviewer and training all see only blurred frames, as in the public dataset; the author found head-blurring barely changes ball detection.

On public match frames it blurs about 1 % of each frame (3 % at most), and about 1 % of labelled balls end up under a blur, typically a ball held at head height. Mark those frames "Exclure image" during review. Extraction takes about 0.3 s per frame. `--blur-conf 0.1` or `--blur-imgsz 1600` blur more aggressively. `--no-blur-heads` turns it off, for footage that has no people. A match folder can't mix blurred and unblurred frames.

**Check it, like a "report a problem" button.** After extraction, a person checks a random sample of each match:

```sh
.venv/bin/python -I teacher/privacy_qa.py sample --match club_20261130 --n 100
# Label Studio: a new project using pseudo/club_20261130/privacy/label_config.xml, import
# pseudo/club_20261130/privacy/tasks.json, box every readable face, head or number plate, export JSON
.venv/bin/python -I teacher/privacy_qa.py apply --match club_20261130 --export ~/Downloads/privacy-export.json
```

`apply` blurs everything you marked, directly in the stored frames, and reports the leak rate. Above 2 %, delete the match's folder and re-extract with `--blur-conf 0.1`, then check again. Use the same process for anyone who asks for more blurring.

**Choosing the sampling rate.** A 90-minute match at 1 frame per second is 5,400 frames. Neighbouring frames are nearly identical, so more frames add review time faster than they add information.
- **Training matches:** `--stride 30` (1 per second). Start with two 15-minute segments if the whole match is too much to review.
- **Reserved matches:** `--stride 60` or more. You review every frame, so 300 to 500 frames is plenty for a test.

**Budget review time.** Expect about 5 to 10 seconds per frame in Label Studio, so 500 frames is about 1 hour. `summary.json` tells you how many frames went to review before you start.

**The labelling rules** (the author's, from `configs/label_studio_all_balls.xml`), in Label Studio (`teacher/label_studio.sh`):
- **One box per visible ball,** including spare balls, stationary balls, balls at the image edges and near the goals, and partly hidden balls.
- **Tight boxes** around the visible ball. For a motion-blurred ball, box the visible blur.
- **No ball visible:** submit with no box. That's a valuable negative.
- **Can't tell** (too dark, too blurred, unsure if it's a ball): **Exclure image**, never a guess.
- **Audit frames** ("audit of an auto-accepted frame"): correct them like any other. They measure how precise auto-accept really is on your footage.

Then import, build and train:

```sh
.venv/bin/python -I teacher/import_review.py --match club_20261130 --export ~/Downloads/project-export.json
.venv/bin/python -I teacher/build_dataset.py --name plus_round1 --matches club_20261130 park_20261207
.venv/bin/python -I train_local.py --data data/plus_round1.yaml --seed 42 --name round1_s42
.venv/bin/python -I train_local.py --data data/plus_round1.yaml --seed 7  --name round1_s7
```

Gate the new models against the public baseline on the original test match **and** the new reserved match, as `MODEL_IMPROVEMENT_PLAN.md` describes. Log every test look in `RESULTS.md`.

## 7. Third-party footage (optional, your call)

The public dataset lists 1,723 annotated frames from 7 public YouTube and TikTok videos (`third_party/annotations.jsonl`, annotations and links only). The author used them for v1.0. Rebuilding them means downloading those videos. The rights stay with their owners: it may be acceptable for local, non-commercial training, but never redistribute the frames. Check each platform's terms. They are mostly broadcast or semi-professional angles, unlike the amateur sideline view, so new matches of your own are worth more.

## 8. Media for the project (demos and sharing)

Separate from training data, short clips help show the app and attract testers. The author's stated bottleneck is real-match testing.

- **What makes a good demo:** 10 to 30 seconds where the camera follows a long ball or a quick switch of play. Add a version with the app's overlays (Réglages avancés → Diagnostic visuel: ball ring, player skeletons) to show what the model sees.
- **Before sharing:** blur faces (the author's demo GIFs blur them), and get the players' consent for public posting, which is a stronger ask than consent for training. Credit the project (box2boxcam, AGPL-3.0, CC BY 4.0 dataset).
- **Where:** a GitHub issue or discussion on the fork, replies in the Reddit thread, or the author's channels if they want it. Always ask the author before posting on their behalf.
- **Useful for the author too:** short notes from each match ("lost the ball on long diagonal passes under floodlights") plus the matching `events.jsonl` excerpt are exactly the real-match feedback they asked for.

## Checklist per match

- [ ] Players told and offered an opt-out (guardians' agreement for minors); venue or league rules checked
- [ ] Filmed with the app at 1080p30; video and logs exported
- [ ] Copied to `~/videos/<match>/`; line added to `matches.csv` with its role
- [ ] `pseudo_label.py` run with de-identification on (train: stride 30, audits on; reserved: stride 60)
- [ ] Privacy check: `privacy_qa.py sample` → review → `apply`, leak rate ≤ 2 %
- [ ] Raw video deleted, or kept off shared and cloud storage
- [ ] Reviewed in Label Studio; export imported
- [ ] Dataset rebuilt; 2 seeds trained; gated against the baseline on both test matches; test look logged
- [ ] Nothing raw published; demo clips de-identified, and players agreed to publication
