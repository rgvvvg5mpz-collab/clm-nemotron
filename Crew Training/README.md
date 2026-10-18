# Crew Training: classifying text with CLM

A 6-minute video plus hands-on exercises. After this session, crew members can start CLM,
ask it questions, train a classifier on labelled data, judge whether it is good enough to
use, and export it to another machine or to S3.

| File | What it is |
| --- | --- |
| [`CLM-Crew-Training.mp4`](CLM-Crew-Training.mp4) | The video: 5:52, 1920×1080 at 30 fps, narrated. Every control is explained with a spotlight and a callout as the tour reaches it; English subtitles built in (switch them on in your player) |
| [`CLM-Crew-Training.srt`](CLM-Crew-Training.srt) | The subtitles as a separate file; also a full transcript |
| Sources | Recorded from the real application by [`Claude Code/tools/make_training_video.py`](../Claude%20Code/tools/make_training_video.py); see "Regenerating the video" below |

Everything in the video is the real application; the model trained on screen is a real
training run on the bundled sample data. Reference material for afterwards:
[the user guide](../docs/USER_GUIDE.md).

## Learning objectives

1. Start, check and stop CLM, with Claude Code or the scripts.
2. Ask yes/no, pick-one and scale questions about a text, and read the answers.
3. Prepare a labelled data file, including when to use **Group by**.
4. Write label descriptions the model can use, and fix weak ones.
5. Train a model and judge it against both baselines, allowing for chance.
6. Use the model from the playground or from code, and know when it is *not* ready for real decisions.
7. Export a model (download, server folder or S3) and load it on another CLM server.

## Chapters

| Time | Chapter |
| ---: | --- |
| 0:00 | Welcome and agenda |
| 0:17 | What CLM is, and what it isn't for |
| 0:45 | Starting the service with Claude Code |
| 1:05 | Playground tour: the two views, the status light, the model list |
| 1:26 | Asking questions: state, question types, model |
| 1:42 | Reading answers; JSON and Python tabs |
| 2:02 | Train a model: the four steps |
| 2:14 | Step 1: preparing your data |
| 2:34 | Step 1: loading the sample, columns, Group by, label counts |
| 2:58 | Step 2: describing labels, and the live hints |
| 3:17 | Step 3: training and the progress chart |
| 3:41 | Step 4: the headline and the two baselines |
| 4:04 | Trying the model on new text |
| 4:12 | Using it: playground, Python, curl |
| 4:26 | Exporting it: download, server folder, S3; the three files |
| 4:58 | The new model in the playground |
| 5:04 | Six habits for your own data |
| 5:28 | Governance: demos versus real decisions |

## Hands-on exercises (about 30 minutes)

Start CLM first (ask Claude **start the service**), then open <http://localhost:8700/>.

**1. Warm up (5 min).** In the playground, load **Examples… → Support ticket triage**,
choose model `typed-decisions-1b` and press **Run**. Change the text to a calm, routine
question and run it again. *What changed in the urgency answer?*

**2. Weak versus strong descriptions (5 min).** Choose model `clm-raw`. In the
`department` question, change the description of `billing` to just "Billing", and run.
Then make it "A charge, invoice, refund or payment problem." and run again. *How did the
probability for billing move?*

**3. Train the sample (10 min).** Go to **Train a model → Try it with sample data**. Make
the `refund` description vague ("Refunds"), name the model `ex3-vague`, and train. Then
restore the description, name it `ex3-clear`, and train again. *Compare the two
accuracies. Which baselines did each beat?*

**4. Export and verify (5 min).** In Step 4 of `ex3-clear`, type `exports/` as the destination
and press **Export**. In a terminal, run `shasum -a 256 -c exports/ex3-clear.sha256`. *Open
`exports/ex3-clear.json`: which encoder does the model need, and what was its accuracy?*

**5. Your own data (10 min).** Make a small CSV (at least 3 labels × 10 rows) from text
you know: email subjects, ticket titles, product reviews. Include a `customer` column if
rows come from the same person. Train it with **Group by** set. *Does it beat "always the
most common label" by more than the "give or take" margin?*

## Knowledge check

1. What does the model actually read for each label: its name or its description?
2. A model scores 72% and "always the most common label" scores 70%, with 100 test
   examples. Is it ready? Why not?
3. Why use **Group by** when several rows come from the same customer?
4. A trained model says 99.9%. What does that tell you, and what doesn't it?
5. When should you use a large language model instead of CLM?
6. What must happen before a model you trained makes real decisions?
7. You export a model and copy it to a server that runs the 8B encoder. What happens, and why?

<details><summary>Answers</summary>

1. Its description. The name is only what you get back.
2. No. A 2-point lead is inside the chance margin (about ±9 points at 100 examples near
   70%). Improve the descriptions or the data first.
3. So near-copies of the same customer's messages can't sit in both training and test,
   which would make the score look better than it really is.
4. It is the model's top choice and a ranking signal. It is not a calibrated probability:
   trained models are over-confident until calibrated.
5. Open questions, writing or summarising, multi-step reasoning, or labels that change on every request.
6. Human-labelled data, calibration, a spot check on real examples, and the new model
   checklist in `Model Governance/`.
7. `clm-serve` refuses to load it: a head only works with the encoder it was trained on, and
   the export's card records which one that was.
</details>

## Facilitator notes

- The sample data is synthetic and labelled by a model. Say so when showing results:
  "about 88%" means agreement with that labeller, not correctness on real customers.
- Retraining on the same data with the same settings gives the same result. Changing the
  descriptions, the data or the seed moves it by a few points; that is expected.
- If a laptop is short of memory, close other apps: the encoder needs about 3 GB.

## Regenerating the video

After a UI change, with CLM running. Reset first so the video shows a first-time training at
real speed (it deletes the previous `ticket-topic` run and the embedding cache):

```bash
bash "Claude Code/scripts/stop.sh"; rm -rf runs/playground/2*-* runs/playground/embeddings exports; bash "Claude Code/scripts/start.sh"
```


```bash
.venv/bin/pip install playwright && .venv/bin/python -m playwright install chromium
```

```bash
.venv/bin/python "Claude Code/tools/make_training_video.py"
```

```bash
.venv/bin/python "Claude Code/tools/make_guide_screenshots.py"
```

The narration uses macOS's built-in speech ("Ava (Premium)" by default, `STUDIO_VOICE` to
change it) and needs `ffmpeg`. The page is captured at 3200×1800 straight from Chrome and
delivered at 1080p; spotlights, callouts and zooms come from `Claude Code/tools/studio.py`.
A run takes about 8 minutes and trains one model. `STUDIO_DRY=1` walks every step in a
minute without speech or frames, to check the script after a UI change.
