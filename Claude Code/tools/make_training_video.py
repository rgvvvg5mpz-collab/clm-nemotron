#!/usr/bin/env python3
"""Record the Crew Training video against a running CLM (start it first with start.sh).

    .venv/bin/python "Claude Code/tools/make_training_video.py"

Writes "Crew Training/CLM-Crew-Training.mp4" (1920x1080, 30 fps, H.264, AAC, English
subtitles) and "Crew Training/CLM-Crew-Training.srt". Everything on screen is the real
application, captured at 3200x1800 and scaled down; the model trained on screen is a real
training run on the bundled sample data. Each control is explained with a spotlight and a
callout as the tour reaches it.

Before recording, reset the server so the video shows a first-time training at real speed:
stop the service, delete the previous run of VIDEO_MODEL_NAME (default ticket-topic) and
the embedding cache from runs/playground/, then start again:

    bash "Claude Code/scripts/stop.sh"; rm -rf runs/playground/2*-* runs/playground/embeddings exports
    bash "Claude Code/scripts/start.sh"
"""
from __future__ import annotations

import os
import sys

from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import studio  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "Crew Training")
BASE = os.environ.get("CLM_URL", "http://127.0.0.1:8700")
MODEL_NAME = os.environ.get("VIDEO_MODEL_NAME", "ticket-topic")
W, H, Z = 1600, 900, 2            # logical page size and the capture zoom (3200x1800 frames)

INK = "#f3f1ef"; MUTED = "#a9a4a0"; RED = "#e2444c"; GOLD = "#e9a200"; TEAL = "#3cc2b4"
FF = "-apple-system,BlinkMacSystemFont,sans-serif"
H1 = f"font:700 58px/1.1 {FF};letter-spacing:-.02em;margin:0 0 18px;color:{INK}"
H2 = f"font:700 42px/1.15 {FF};margin:0 0 26px;color:{INK}"
P = f"font:400 26px/1.5 {FF};color:{MUTED};margin:0"
EYEBROW = f"font:700 16px {FF};letter-spacing:.14em;text-transform:uppercase;color:{RED};margin:0 0 18px"
TERM = "font:400 21px/1.6 ui-monospace,Menlo,monospace"


def li(items, color=INK):
    return "".join(f'<div style="display:flex;gap:16px;align-items:baseline;margin:0 0 14px;font:400 27px/1.4 {FF};color:{color}">'
                   f'<span style="color:{RED};font-weight:700">{i + 1:02d}</span><span>{t}</span></div>' for i, t in enumerate(items))


# ----------------------------------------------------------------------------- cards
CARD_TITLE = f"""<p style="{EYEBROW}">Crew training · 6 minutes</p>
<h1 style="{H1}">Classify text with your own labels</h1>
<p style="{P};margin-bottom:46px">Start CLM, ask it questions, train a classifier, judge it, and take it with you.</p>
{li(["Start the service", "Tour the playground", "Prepare your data", "Train, step by step", "Read the results, use and export the model"])}"""

CARD_WHAT = f"""<p style="{EYEBROW}">What CLM is</p>
<h2 style="{H2}">A frozen reader, two small learners, one fast answer</h2>
<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:18px;margin-bottom:40px">
 <div style="border:1px solid #333;border-radius:14px;padding:22px"><div style="font:700 23px {FF};color:{GOLD}">1 · Nemotron reads</div><p style="{P};font-size:21px;margin-top:10px">An NVIDIA embedding model turns each text into numbers. It never changes.</p></div>
 <div style="border:1px solid #333;border-radius:14px;padding:22px"><div style="font:700 23px {FF};color:{TEAL}">2 · Heads learn</div><p style="{P};font-size:21px;margin-top:10px">Two small networks learn your labels from your examples, in seconds to minutes.</p></div>
 <div style="border:1px solid #333;border-radius:14px;padding:22px"><div style="font:700 23px {FF};color:{RED}">3 · You get odds</div><p style="{P};font-size:21px;margin-top:10px">A probability for every label, in tens of milliseconds, on your own machine.</p></div>
</div>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:30px;font:400 23px/1.5 {FF}">
 <div><b style="color:{TEAL}">Good for</b><div style="color:{INK}">routing, tagging, triage, yes/no checks with fixed labels and examples</div></div>
 <div><b style="color:{RED}">Not for</b><div style="color:{INK}">open questions, writing, multi-step reasoning: use a large language model</div></div>
</div>"""

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
<p style="{P};font-size:22px;margin-top:22px">First run: installs and downloads the models (a few minutes). After that: seconds. Ask "is it running?" or "stop the service" any time.</p>"""

CARD_DATA = f"""<p style="{EYEBROW}">Step 1 · Your data</p>
<h2 style="{H2}">One column of text, one column with the right label</h2>
<div style="background:#000;border:1px solid #333;border-radius:14px;padding:20px 26px;{TERM};font-size:20px;color:#d6d6d6;margin-bottom:30px">
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
<p style="{P};font-size:22px;margin-top:30px">Step-by-step: <span style="color:{GOLD}">docs/USER_GUIDE.md</span> · Governance: <span style="color:{GOLD}">Model Governance/</span></p>"""


# ----------------------------------------------------------------------------- narration
N = {
    "intro": "Welcome to C L M crew training. In about six minutes you'll learn to start the service, ask it "
             "questions, and train a classifier on your own labelled examples. You'll see how to judge whether it is "
             "good enough to use, and how to export it to another machine or to S3.",
    "what": "C L M is a fast classifier. A frozen NVIDIA Nemotron model reads each text once and turns it into "
            "numbers. Two small networks on top learn your labels. Training takes seconds to minutes, answers come back "
            "in tens of milliseconds, and nothing leaves your machine. Use it for routing, tagging and triage, where the "
            "labels are fixed and you have examples. For open questions or writing, use a large language model instead.",
    "start": "To start it, clone the repository, open Claude Code in the folder, and type: start the service. "
             "The first time, it installs everything and downloads the models, which takes a few minutes. After that it "
             "starts in seconds, and prints the address to open: localhost, port eighty-seven hundred.",
    "tour": "This is the playground. At the top, a switch between asking questions and training a model. On the "
            "right, the status light: green means the service and its encoder are up. Click it to see every loaded "
            "model, what each one answers, how accurate it was on its test set, and which one is in use. The arrow "
            "beside a model downloads it.",
    "ask": "On the left is the state: the text you want to classify. Below it are the questions. A yes or no "
           "question, a pick-one question with its options described in plain words, and a rating scale. "
           "Choose a model, here the shipped typed decisions model, and press Run.",
    "answers": "Each question gets its own card. The highlighted bar is the model's answer, and the percentage is "
               "its probability. Treat those numbers as a ranking, not a guarantee, until the model has been "
               "calibrated. The J SON tab shows exactly what came back, and the Python tab gives a developer the "
               "same request as code.",
    "train": "Now the main event: training your own classifier. Switch to Train a model. Four steps on one page: "
             "add examples, describe the labels, train, and check the results.",
    "data": "First, the data. You need a file with one column of text and one column with the correct label. "
            "A spreadsheet saved as C S V works. Aim for at least thirty examples per label. If several rows come from "
            "the same customer or case, include that I D column: you'll use it to keep the test honest.",
    "sample": "We'll use the bundled sample: four hundred synthetic support conversations in five categories. "
              "C L M finds the text and label columns by itself; you can change them here. Group by keeps one "
              "customer's rows together. The preview highlights the two columns, and the chips show how many "
              "examples each label has. A label with very few examples gets a warning.",
    "describe": "Step two matters most. The model doesn't read your label names. It reads your descriptions, and "
                "compares every text against them. So describe each label with the words that actually appear in its "
                "examples. Watch what happens with a weak description: the page warns you as you type. Let's put the "
                "good one back.",
    "go": "Step three. Give the model a name, and check how the data is split, here the file's own test rows. "
          "Press Train model. The encoder reads every text once, then the small networks train in rounds. The line "
          "is accuracy on a check set the model never trains on, and the dashed line is where it started. Training "
          "stops by itself when it stops improving.",
    "training": "While it trains, each point on the line is one round. The gap between the line and the dashed "
                "line is what your examples taught it. If the line stops climbing, the page stops early and keeps "
                "the best round.",
    "results": "Step four: is it any good? The headline is how many of a hundred unseen conversations it got right. "
               "Below it, two baselines: the base model without any training, and simply always guessing the most "
               "common label. A useful model must clearly beat both. And mind the margin: with a hundred test "
               "examples, a few points either way is just chance.",
    "try": "Try it on a new message before you trust it. Paste any text and press Classify. Here, a double charge "
           "is routed to billing.",
    "use": "The model is already live on this server. Open in playground loads it with its question, and the copy "
           "buttons give developers ready-made Python or curl. It is saved, so it comes back after a restart.",
    "export": "To take the model somewhere else, export it. Download gives you one archive for this computer. Or "
              "type a folder on the server, or an S3 location like s3 colon slash slash my bucket, and press Export. "
              "You get three files: the model, a card that records its encoder, its labels and its accuracy, and "
              "checksums. Another C L M server loads the model with one flag, as long as it runs the same encoder. "
              "From Claude Code, just say: export ticket topic to S3.",
    "open": "Back in the playground, the new model answers the same question on any text you give it.",
    "habits": "With your own data, a few habits matter most. Use labels that people have checked. Group rows by "
              "customer, so the test stays honest. Describe each label with the words its examples use, and avoid "
              "catch-alls like other. If the model doesn't clearly beat both baselines, improve the descriptions or the "
              "data before anything else. And retrain whenever your labels change.",
    "governance": "Finally, governance. The models shipped with the repository are demos, trained on synthetic "
                  "data. Before a model makes real decisions, it needs human-labelled data, calibrated probabilities, "
                  "and the new model checklist in the Model Governance folder. The user guide covers everything in this "
                  "video, step by step. Thanks for watching.",
}


def main():
    work = studio.mkwork("clm-video-")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": W * Z, "height": H * Z}, device_scale_factor=1)
        page = ctx.new_page()
        page.goto(BASE + "/")
        page.wait_for_selector("#status.ok, #status.warn", timeout=30000)
        page.evaluate("() => { try { localStorage.clear(); } catch (e) {} }")       # a fresh visitor's playground
        page.reload()
        page.wait_for_selector("#status.ok, #status.warn", timeout=30000)
        rec = studio.Recorder(ctx, page, work, N, logical=(W, H), zoom=Z)
        page.wait_for_timeout(600)

        # 1-3 · cards
        rec.card(CARD_TITLE); rec.scene("intro"); rec.hold(0.8)
        rec.card(CARD_WHAT); rec.scene("what"); rec.hold(0.8)
        rec.card(CARD_START); rec.scene("start"); rec.hold(0.8)
        rec.card(None); rec.wait(0.6)

        # 4 · tour: views, status light, models panel
        rec.scene("tour", "The playground", "Tour")
        rec.spot(".viewswitch", "Two views", "Playground: ask questions about a text. Train a model: teach it your own labels.", wait=4.2)
        rec.spot("#status", "Status light", "Green: the service and its encoder are up. The number is how many models are loaded. Click it for the list.", wait=3.6)
        rec.click("#status", wait=0.9)
        rec.spot("#models-panel", "Loaded models", "What each model answers, its test accuracy, which is the default and which is in use. Click one to use it; ↓ downloads it.", wait=6.5)
        rec.hold()
        page.keyboard.press("Escape"); rec.unspot(); rec.wait(0.4)

        # 5-6 · ask and read answers
        rec.scene("ask", "Ask questions about a text", "Tour")
        rec.spot("#state", "State", "The text to classify: a message, a ticket, a document. Up to 8,000 tokens.", wait=3.4)
        rec.spot(".q-card", "A question", "noul = yes/no. choice = pick one option; each option is described in plain words. score = a scale.", wait=6.0)
        rec.spot("#model", "Model", "Which trained model answers. clm-raw is the untrained baseline.", wait=2.4)
        page.select_option("#model", "typed-decisions-1b"); rec.wait(0.5)
        rec.unspot()
        rec.click("#run", wait=0.3)
        page.wait_for_selector("#view-answer .ans", timeout=20000); rec.wait(1.0)
        rec.hold()
        rec.scene("answers", "Read the answers", "Tour")
        rec.spot("#view-answer .ans", "Answer card", "The highlighted bar is the answer; the percentage is the model's probability. A ranking, not a guarantee, until calibrated.", wait=7.0)
        rec.unspot()
        rec.click('.pane-res .tab[data-view="json"]', wait=0.6)
        rec.spot("#view-json", "JSON", "Exactly what the API returned: every probability, the model, token counts.", wait=3.2)
        rec.unspot()
        rec.click('.pane-res .tab[data-view="python"]', wait=0.6)
        rec.spot("#view-python", "Python", "The same request as code, ready to paste into an application.", wait=3.0)
        rec.unspot()
        rec.click('.pane-res .tab[data-view="answer"]')
        rec.hold()

        # 7-9 · train: four steps, data
        rec.scene("train", "Train a model", "Train")
        rec.click('.vs-btn[data-view="train"]', wait=1.0)
        for i, (t, d) in enumerate([("Step 1", "Add a file of labelled examples."), ("Step 2", "Describe what each label means."),
                                    ("Step 3", "Train, and watch it learn."), ("Step 4", "Check the results, use and export the model.")], 1):
            rec.spot(f"#t-step-{i} .step-title", t, d, wait=1.6, zoom=1)
        rec.unspot()
        rec.hold()
        rec.card(CARD_DATA); rec.scene("data"); rec.hold(0.8); rec.card(None); rec.wait(0.5)

        rec.scene("sample", "Step 1: add labelled examples", "Step 1 of 4")
        rec.click("#t-sample", wait=2.0)
        page.wait_for_selector("#t-data .file-chip")
        rec.spot("#t-text-col", "Text to classify", "The column the model reads. Found automatically; change it if it guessed wrong.", wait=3.2)
        rec.spot("#t-label-col", "Correct label", "The column with the right answer. Every different value becomes a label.", wait=3.0)
        rec.spot("#t-group-col", "Group by", "Pick the customer or case column so one customer's rows never straddle training and test.", wait=3.6)
        rec.spot("#t-data .t-table-wrap", "Preview", "Teal: the text column. Red: the label column.", wait=3.0)
        rec.spot("#t-data .chips", "Examples per label", "A label with very few examples is flagged here.", wait=3.0)
        rec.hold()
        rec.unspot()

        # 10 · describe labels
        rec.scene("describe", "Step 2: describe what each label means", "Step 2 of 4")
        rec.scroll_to("#t-step-2")
        rec.spot("#t-q", "The question", "Asked of every text. The model compares the text with each description below.", wait=4.0)
        good = page.input_value("#t-opt-refund")
        rec.spot("#t-opt-refund", "A label's description", "This is what the model actually reads for “refund”. Use the words its examples use.", wait=3.0)
        rec.type("#t-opt-refund", "Refunds", delay=90)
        rec.spot("#t-hint-refund", "Live hint", "Too short or too generic, and the page says so while you type.", wait=3.2)
        rec.spot("#t-opt-refund", "", "", wait=0.2)
        rec.type("#t-opt-refund", good, delay=22)
        rec.hold()
        rec.unspot()

        # 11 · train
        rec.scene("go", "Step 3: train", "Step 3 of 4")
        rec.scroll_to("#t-step-3")
        rec.spot("#t-name", "Model name", "How the model appears in the playground and the API.", wait=0.8)
        rec.type("#t-name", MODEL_NAME, delay=60)
        rec.spot(".split-line", "The split", "Rows marked “test” are held back and never trained on; the score comes from them.", wait=3.6)
        rec.unspot()
        rec.click("#t-go", wait=1.2)
        rec.spot("#t-progress", "Progress", "Solid line: accuracy on the check set after each round. Dashed: where it started. It stops when it stops improving.", wait=1.0)
        rec.hold(0.3)
        if page.locator("#t-step-4.is-locked").count():          # still training: say what the chart shows
            rec.scene("training", "Step 3: watching it learn", "Step 3 of 4")
            rec.hold(0.3)
        page.wait_for_selector("#t-step-4:not(.is-locked)", timeout=600000)
        rec.wait(1.2)
        rec.unspot()

        # 12-16 · results, try, use, export, open
        rec.scene("results", "Step 4: check the results", "Step 4 of 4")
        rec.scroll_to("#t-step-4")
        rec.spot("#t-results .verdict", "The headline", "Accuracy on held-out examples, with the margin that chance alone can move it by.", wait=6.5)
        rec.spot("#t-results .cmp", "Two baselines", "The encoder with no training, and always guessing the commonest label. Beat both, clearly.", wait=7.0)
        rec.hold()
        rec.unspot()
        rec.scene("try", "Try it on new text", "Step 4 of 4")
        rec.spot("#t-try", "Try it", "Paste any new text. Nothing typed here is stored.", wait=2.4)
        rec.unspot()
        rec.click("#t-try-go", wait=0.3)
        page.wait_for_selector("#t-try-out .ans, #t-try-out .t-err", timeout=20000); rec.wait(1.2)
        rec.spot("#t-try-out", "The answer", "Routed to billing, with the model's probability for every label.", wait=2.0)
        rec.hold()
        rec.unspot()
        rec.scene("use", "Use it", "Step 4 of 4")
        rec.spot(".use .actions", "Use it", "Open in playground loads the model with its question. Copy Python and Copy curl give working code.", wait=5.0)
        rec.hold()
        rec.unspot()
        rec.scene("export", "Export it: to disk or to S3", "Step 4 of 4")
        rec.spot("#t-download", "Download", "One .tar.gz for this computer: the model, its card and checksums.", wait=3.6)
        rec.spot("#t-export-dest", "Folder or S3", "A folder on the server, or s3://bucket/prefix/. S3 uses the server's AWS credentials.", wait=2.2)
        rec.type("#t-export-dest", "exports/", delay=70)
        rec.unspot()
        rec.click("#t-export-go", wait=0.3)
        page.wait_for_selector("#t-export-out .export-done, #t-export-out .t-err", timeout=30000); rec.wait(1.5)
        rec.spot("#t-export-out", "Three files", "name.pt is the model. name.json records the encoder, labels and accuracy. name.sha256 verifies the copy. The last line loads it on another server.", wait=9.0)
        rec.hold()
        rec.unspot()
        rec.scene("open", "The new model in the playground", "Step 4 of 4")
        rec.click("#t-open", wait=1.5)
        rec.click("#run", wait=0.3)
        page.wait_for_selector("#view-answer .ans", timeout=20000); rec.wait(1.2)
        rec.spot("#view-answer .ans", "", "", wait=2.0)
        rec.hold()
        rec.unspot()

        # 17-18 · habits, governance
        rec.card(CARD_HABITS); rec.scene("habits"); rec.hold(0.8)
        rec.card(CARD_GOV); rec.scene("governance"); rec.hold(1.5)
        end = rec.finish()
        browser.close()

    srt = os.path.join(OUT_DIR, "CLM-Crew-Training.srt")
    studio.write_srt(rec.scenes, srt)
    out = os.path.join(OUT_DIR, "CLM-Crew-Training.mp4")
    studio.assemble(rec, out, srt, end)
    studio.cleanup(work)
    chapters = "\n".join(f"{int(s['start'] // 60)}:{int(s['start'] % 60):02d}  {s['key']}" for s in rec.scenes)
    print(f"wrote {out} ({end / 60:.1f} min) and {srt}\n{chapters}")


if __name__ == "__main__":
    main()
