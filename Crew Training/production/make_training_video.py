#!/usr/bin/env python3
"""Record the Crew Training video against a running CLM (start it first with start.sh).

    .venv/bin/python "Crew Training/production/make_training_video.py"

Writes "Crew Training/CLM-Crew-Training.mp4" (1280x800, H.264, AAC, English subtitles) and
"Crew Training/CLM-Crew-Training.srt". Everything on screen is the real application; the
model trained in the video is a real training run on the bundled sample data.
"""
from __future__ import annotations

import glob
import os
import shutil
import sys
import tempfile

from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import studio  # noqa: E402

OUT_DIR = os.path.dirname(HERE)
BASE = os.environ.get("CLM_URL", "http://127.0.0.1:8700")
MODEL_NAME = os.environ.get("VIDEO_MODEL_NAME", "ticket-topic")

INK = "#f3f1ef"; MUTED = "#a9a4a0"; RED = "#e2444c"; GOLD = "#e9a200"; TEAL = "#3cc2b4"
H1 = f"font:700 54px/1.1 -apple-system,BlinkMacSystemFont,sans-serif;letter-spacing:-.02em;margin:0 0 18px;color:{INK}"
H2 = f"font:700 40px/1.15 -apple-system,BlinkMacSystemFont,sans-serif;margin:0 0 26px;color:{INK}"
P = f"font:400 25px/1.5 -apple-system,BlinkMacSystemFont,sans-serif;color:{MUTED};margin:0"
EYEBROW = f"font:700 16px -apple-system,sans-serif;letter-spacing:.14em;text-transform:uppercase;color:{RED};margin:0 0 18px"


def li(items, color=INK):
    return "".join(f'<div style="display:flex;gap:16px;align-items:baseline;margin:0 0 14px;font:400 26px/1.4 -apple-system,sans-serif;color:{color}">'
                   f'<span style="color:{RED};font-weight:700">{i + 1:02d}</span><span>{t}</span></div>' for i, t in enumerate(items))


CARD_TITLE = f"""<p style="{EYEBROW}">Crew training · 5 minutes</p>
<h1 style="{H1}">Classify text with your own labels</h1>
<p style="{P};margin-bottom:46px">Start CLM, ask it questions, train a classifier and decide if it is good enough to use.</p>
{li(["Start the service", "Tour the playground", "Prepare your data", "Train, step by step", "Read the results and use the model"])}"""

CARD_WHAT = f"""<p style="{EYEBROW}">What CLM is</p>
<h2 style="{H2}">A frozen reader, two small learners, one fast answer</h2>
<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:18px;margin-bottom:40px">
 <div style="border:1px solid #333;border-radius:14px;padding:22px"><div style="font:700 22px sans-serif;color:{GOLD}">1 · Nemotron reads</div><p style="{P};font-size:20px;margin-top:10px">An NVIDIA embedding model turns each text into numbers. It never changes.</p></div>
 <div style="border:1px solid #333;border-radius:14px;padding:22px"><div style="font:700 22px sans-serif;color:{TEAL}">2 · Heads learn</div><p style="{P};font-size:20px;margin-top:10px">Two small networks learn your labels from your examples, in seconds to minutes.</p></div>
 <div style="border:1px solid #333;border-radius:14px;padding:22px"><div style="font:700 22px sans-serif;color:{RED}">3 · You get odds</div><p style="{P};font-size:20px;margin-top:10px">A probability for every label, in tens of milliseconds, on your own machine.</p></div>
</div>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:30px;font:400 22px/1.5 sans-serif">
 <div><b style="color:{TEAL}">Good for</b><div style="color:{INK}">routing, tagging, triage, yes/no checks with fixed labels and examples</div></div>
 <div><b style="color:{RED}">Not for</b><div style="color:{INK}">open questions, writing, multi-step reasoning: use a large language model</div></div>
</div>"""

TERM = "font:400 21px/1.6 ui-monospace,Menlo,monospace"
CARD_START = f"""<p style="{EYEBROW}">Start the service</p>
<h2 style="{H2}">Clone, open Claude Code, ask</h2>
<div style="background:#000;border:1px solid #333;border-radius:14px;padding:26px 30px;{TERM};color:#d6d6d6">
<div><span style="color:{TEAL}">$</span> git clone https://github.com/rgvvvg5mpz-collab/clm-nemotron.git</div>
<div><span style="color:{TEAL}">$</span> cd clm-nemotron &amp;&amp; claude</div>
<div style="margin-top:10px"><span style="color:{GOLD}">&gt;</span> start the service</div>
<div style="color:{MUTED};margin-top:10px">[clm] starting the nemotron-3-embed-1b encoder on :8090 (mac)</div>
<div style="color:{MUTED}">[clm] starting clm-serve on :8700</div>
<div style="color:#fff">[clm] running. Open <span style="color:{GOLD}">http://localhost:8700/</span>  (models: support-topic-1b, typed-decisions-1b, clm-raw)</div>
</div>
<p style="{P};font-size:21px;margin-top:22px">First run: installs and downloads the models (a few minutes). After that: seconds. Ask "is it running?" or "stop the service" any time.</p>"""

CARD_DATA = f"""<p style="{EYEBROW}">Step 1 · Your data</p>
<h2 style="{H2}">One column of text, one column with the right label</h2>
<div style="background:#000;border:1px solid #333;border-radius:14px;padding:20px 26px;{TERM};font-size:19px;color:#d6d6d6;margin-bottom:30px">
<div style="color:{GOLD}">text,label,customer</div>
<div>"I was charged twice for March",billing,C-1042</div>
<div>"The app crashes when I log in",technical,C-0007</div>
<div>"My parcel never arrived",delivery,C-0311</div>
</div>
{li(["CSV, JSONL, JSON or Parquet: a spreadsheet saved as CSV is fine",
     "At least 30 examples per label, labels checked by people",
     "Add the customer or case ID if rows belong together"])}"""

CARD_HABITS = f"""<p style="{EYEBROW}">With your own data</p>
<h2 style="{H2}">Six habits that decide accuracy</h2>
{li(["Labels checked by people, not guessed by a model",
     "Group by customer or case, so the test stays honest",
     "Describe each label with the words its examples use",
     "No catch-alls like “other”: they match everything a little",
     "Must clearly beat both baselines, or fix descriptions and data first",
     "Retrain when labels, wording or sources change"])}"""

CARD_GOV = f"""<p style="{EYEBROW}">Before real decisions</p>
<h2 style="{H2}">Demo models are demos</h2>
<p style="{P};margin-bottom:30px">The shipped models were trained on synthetic data labelled by a model. A model you train makes real decisions only after:</p>
{li(["Human-labelled training and test data",
     "Calibration of its probabilities, and thresholds set from validation",
     "The new model checklist in the Model Governance folder"])}
<p style="{P};font-size:21px;margin-top:30px">Step-by-step: <span style="color:{GOLD}">docs/USER_GUIDE.md</span> · Governance: <span style="color:{GOLD}">Model Governance/</span></p>"""


def main():
    work = tempfile.mkdtemp(prefix="clm-video-")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=1,
                                  record_video_dir=work, record_video_size={"width": 1440, "height": 900})
        page = ctx.new_page()
        rec = studio.Recorder(page, work)
        page.goto(BASE + "/")
        page.wait_for_selector("#status.ok, #status.warn", timeout=30000)
        page.wait_for_timeout(800)

        # 1-3 · cards
        rec.card(CARD_TITLE)
        rec.scene("intro", "Welcome to C L M crew training. In about five minutes you'll learn to start the service, "
                  "ask it questions, and train a classifier on your own labelled examples. Then you'll see how to judge "
                  "whether that classifier is good enough to use.", "")
        rec.hold(0.8)
        rec.card(CARD_WHAT)
        rec.scene("what", "C L M is a fast classifier. A frozen NVIDIA Nemotron model reads each text once and turns it into "
                  "numbers. Two small networks on top learn your labels. Training takes seconds to minutes, answers come back "
                  "in tens of milliseconds, and nothing leaves your machine. Use it for routing, tagging and triage, where the "
                  "labels are fixed and you have examples. For open questions or writing, use a large language model instead.",
                  "")
        rec.hold(0.8)
        rec.card(CARD_START)
        rec.scene("start", "To start it, clone the repository, open Claude Code in the folder, and type: start the service. "
                  "The first time, it installs everything and downloads the models, which takes a few minutes. After that it "
                  "starts in seconds, and prints the address to open: localhost, port eighty-seven hundred.", "")
        rec.hold(0.8)
        rec.card(None)

        # 4-6 · playground tour
        rec.scene("tour", "This is the playground. At the top: the switch between the playground and training, and the status "
                  "light, which shows that the service is ready, and how many models are loaded. Hover over any question mark "
                  "for an explanation.", "The playground", "Tour")
        rec.point(".viewswitch"); rec.point("#status")
        rec.point('label[for="state"] + .tip-q')
        page.hover('label[for="state"] + .tip-q'); page.wait_for_timeout(1600)
        rec.hold()
        page.mouse.move(700, 600)
        rec.scene("ask", "On the left is the state: the text you want to classify. Below it are the questions. A yes or no "
                  "question, a pick-one question with options, and a rating scale. Choose a model, here the shipped typed "
                  "decisions model, and press Run.", "Ask questions about a text", "Tour")
        rec.point("#state"); rec.point(".q-card"); rec.point("#model")
        page.select_option("#model", "typed-decisions-1b"); page.wait_for_timeout(500)
        rec.click("#run", wait=1.2)
        rec.hold()
        rec.scene("answers", "Each question gets its own chart. The highlighted bar is the model's answer, and the percentage "
                  "is its probability. Treat those numbers as a ranking, not a guarantee, until the model has been calibrated. "
                  "The JSON tab shows exactly what came back, and the Python tab gives a developer the same request as code.",
                  "Read the answers", "Tour")
        rec.point("#view-answer .ans", wait=2.5)
        rec.click('.pane-res .tab[data-view="json"]', wait=1.8)
        rec.click('.pane-res .tab[data-view="python"]', wait=1.8)
        rec.click('.pane-res .tab[data-view="answer"]')
        rec.hold()

        # 7-9 · train: data
        rec.scene("train", "Now the main event: training your own classifier. Switch to Train a model. Four steps on one page: "
                  "add examples, describe the labels, train, and check the results.", "Train a model", "Train")
        rec.click('.vs-btn[data-view="train"]', wait=1.0)
        for i in (1, 2, 3, 4):
            rec.point(f"#t-step-{i} .step-title", wait=0.6)
        rec.hold()
        rec.card(CARD_DATA)
        rec.scene("data", "First, the data. You need a file with one column of text and one column with the correct label. "
                  "A spreadsheet saved as C S V works. Aim for at least thirty examples per label. If several rows come from "
                  "the same customer or case, include that I D column: you'll use it to keep the test honest.", "")
        rec.hold(0.8)
        rec.card(None)
        rec.scene("sample", "We'll use the bundled sample: four hundred synthetic support conversations, in five categories. "
                  "C L M finds the text and label columns by itself, and highlights them in the preview. The chips show how "
                  "many examples each label has. A label with very few examples gets a warning.",
                  "Step 1: add labelled examples", "Step 1 of 4")
        rec.click("#t-sample", wait=2.0)
        page.wait_for_selector("#t-data .file-chip")
        rec.point("#t-text-col"); rec.point("#t-label-col")
        rec.point("#t-data .t-table-wrap", wait=1.2)
        rec.point("#t-data .chips", wait=1.2)
        rec.hold()

        # 10 · describe labels
        rec.scene("describe", "Step two matters most. The model doesn't read your label names. It reads your descriptions, "
                  "and compares every text against them. So describe each label with the words that actually appear in its "
                  "examples. Watch what happens with a weak description: the page warns you as you type. Let's put the good "
                  "one back.", "Step 2: describe what each label means", "Step 2 of 4")
        rec.scroll_to("#t-step-2")
        rec.point("#t-q", wait=1.0)
        good = page.input_value("#t-opt-refund")
        rec.type("#t-opt-refund", "Refunds", delay=90)
        rec.point("#t-hint-refund", wait=2.5)
        rec.type("#t-opt-refund", good, delay=22)
        rec.hold()

        # 11 · train
        rec.scene("go", "Step three. Give the model a name, check how the data is split, here the file's own test rows, and "
                  "press Train model. The encoder reads every text once, then the small networks train in rounds. The line "
                  "shows accuracy on a check set the model doesn't train on. The dashed line is where it started, without "
                  "training. Training stops by itself when it stops improving.", "Step 3: train", "Step 3 of 4")
        rec.scroll_to("#t-step-3")
        rec.type("#t-name", MODEL_NAME, delay=60)
        rec.point(".split-line", wait=1.0)
        rec.click("#t-go", wait=1.0)
        rec.point("#t-progress", ring=False, wait=0.5)
        page.wait_for_selector("#t-step-4:not(.is-locked)", timeout=600000)
        rec.hold(0.3)

        # 12-14 · results, try, use
        rec.scene("results", "Step four: is it any good? This model gets most of a hundred unseen conversations right. "
                  "Compare that with the two baselines below it: the base model without training, and simply always guessing "
                  "the most common label. A useful model must clearly beat both. And mind the margin: with a hundred test "
                  "examples, a few points either way is just chance.", "Step 4: check the results", "Step 4 of 4")
        rec.scroll_to("#t-step-4")
        rec.point("#t-results .verdict", wait=2.0)
        rec.point("#t-results .cmp", wait=2.0)
        rec.hold()
        rec.scene("try", "Try it on a new message before you trust it. Type or paste any text, and press Classify. "
                  "Here, a double charge is routed to billing.", "Try it on new text", "Step 4 of 4")
        rec.point("#t-try", wait=1.0)
        rec.click("#t-try-go", wait=1.5)
        rec.point("#t-try-out", wait=2.0)
        rec.hold()
        rec.scene("use", "The model is already live on this server. The copy buttons give developers ready-made Python or "
                  "curl. Open in playground loads the model with its question, ready to use. And it's saved, so it comes back "
                  "after a restart.", "Use it", "Step 4 of 4")
        rec.point("#t-copy-py", wait=1.2)
        rec.click("#t-open", wait=1.5)
        rec.click("#run", wait=1.5)
        rec.point("#view-answer .ans", wait=1.5)
        rec.hold()

        # 15-16 · habits, governance
        rec.card(CARD_HABITS)
        rec.scene("habits", "With your own data, a few habits matter most. Use labels that people have checked. Group rows by "
                  "customer, so the test stays honest. Describe each label with the words its examples use, and avoid "
                  "catch-alls like other. If the model doesn't clearly beat both baselines, improve the descriptions or the "
                  "data before anything else. And retrain whenever your labels change.", "")
        rec.hold(0.8)
        rec.card(CARD_GOV)
        rec.scene("governance", "Finally, governance. The models shipped with the repository are demos, trained on synthetic "
                  "data. Before a model makes real decisions, it needs human-labelled data, calibrated probabilities, and the "
                  "new model checklist in the Model Governance folder. The user guide covers everything in this video, step by "
                  "step. Thanks for watching.", "")
        rec.hold(1.5)
        end = rec.now()

        page.close(); ctx.close(); browser.close()

    video = glob.glob(os.path.join(work, "*.webm"))[0]
    srt = os.path.join(OUT_DIR, "CLM-Crew-Training.srt")
    studio.write_srt(rec.scenes, srt)
    out = os.path.join(OUT_DIR, "CLM-Crew-Training.mp4")
    studio.assemble(video, rec.scenes, out, srt, end=end)
    shutil.rmtree(work, ignore_errors=True)
    print(f"wrote {out} ({end / 60:.1f} min) and {srt}")


if __name__ == "__main__":
    main()
