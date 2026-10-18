#!/usr/bin/env python3
"""Build the executive briefing: slides (PNG, PDF), an editable PowerPoint, and a ~3-minute video.

    .venv/bin/python Marketing/production/build_marketing.py

Inputs: Marketing/benchmark/results.json (CLM measured by evaluation/benchmark_service.py)
and, when present, Marketing/benchmark/frontier-results.json (from compare_frontier.py).
Frontier accuracy and latency appear only if that second file exists; costs otherwise come
from list prices with the token assumptions printed on the slide. Needs Playwright (slide
rendering), python-pptx, macOS `say` and ffmpeg.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
MKT = os.path.dirname(HERE)
REPO = os.path.dirname(MKT)
sys.path.insert(0, os.path.join(REPO, "Crew Training", "production"))
import studio  # noqa: E402

# ----------------------------------------------------------------------------- numbers
R = json.load(open(os.path.join(MKT, "benchmark", "results.json")))
F_PATH = os.path.join(MKT, "benchmark", "frontier-results.json")
FRONTIER = json.load(open(F_PATH))["results"] if os.path.exists(F_PATH) else None
CLM = R["support-topic-1b"]
P50, P95 = round(CLM["latency_ms_first_time"]["p50"]), round(CLM["latency_ms_first_time"]["p95"])
RPS = R["throughput"]["requests_per_second"]
PER_DAY_M = RPS * 86400 / 1e6
HOURS_PER_M = 1e6 / RPS / 3600
COMPUTE_RATE = 1.00                                  # USD per compute-hour: an assumption, stated on the slide
CLM_COST = HOURS_PER_M * COMPUTE_RATE
TOK_IN, TOK_OUT = 330, 50                            # estimated tokens per frontier request (stated on the slide)
PRICES = {"Haiku 5.5": (0.10, 0.50), "Sonnet 5.5": (2.00, 10.00), "Opus 5.5": (4.00, 20.00)}
COSTS = {m: (TOK_IN * i + TOK_OUT * o) for m, (i, o) in PRICES.items()}   # USD per 1M requests
ACC_SHIPPED, ACC_LO, ACC_HI, ACC_MEAN = 88, 88, 96, 92
ACC_RAW, ACC_MAJ = round(R["clm-raw"]["accuracy"] * 100), 18

INK, MUTED, LINE, BG = "#10213a", "#5a6476", "#dfe3ea", "#ffffff"
FF = '-apple-system,BlinkMacSystemFont,"Helvetica Neue",Arial,sans-serif'
TEAL, ORANGE, GREY = "#0f7b6c", "#d9622b", "#b9c0cb"

# ----------------------------------------------------------------------------- slides
SLIDES = [
    {"key": "title", "kicker": "Executive briefing",
     "title": "Fast, cheap, private text classification",
     "sub": "When a small trained model beats calling a frontier language model",
     "notes": "Every day our teams route tickets, tag documents and triage alerts, and many now send each decision to a frontier language model. Here's when a small trained classifier does that job better: cheaper, faster, and inside our walls."},
    {"key": "problem", "kicker": "The job",
     "title": "High-volume classification is a different job",
     "bullets": ["Millions of short, repetitive decisions: route, tag, flag",
                 "A fixed set of labels, and past examples of each",
                 "Needs to be fast, cheap per decision, private and consistent",
                 "A frontier model is a generalist, paid per token, run remotely"],
     "notes": 'Classification at volume is millions of short, repetitive decisions with fixed labels and plenty of past examples. What matters is speed, cost per decision, privacy and consistency. A frontier model is a brilliant generalist, but we pay per token, and every request leaves the building.'},
    {"key": "what", "kicker": "The approach",
     "title": "A frozen reader, a small learner, one fast answer",
     "steps": [("NVIDIA Nemotron reads", "Turns each text into numbers. Never retrained."),
               ("A small model learns", "Our labels, from our examples, in seconds to minutes."),
               ("We get the odds", "A probability per label, in milliseconds, on our hardware.")],
     "notes": 'C L M splits the work. An NVIDIA Nemotron model reads each text and never changes. A small model on top learns our labels from our own examples, in minutes, from a browser. We get a probability for every label, in milliseconds, on our own hardware.'},
    {"key": "accuracy", "kicker": "Accuracy (measured)",
     "title": f"Trained on 300 examples: {ACC_SHIPPED}% to {ACC_HI}% right",
     "bars": [("Trained CLM model", ACC_SHIPPED, TEAL, f"{ACC_SHIPPED}–{ACC_HI}%"),
              ("Same reader, no training", ACC_RAW, GREY, f"{ACC_RAW}%"),
              ("Always guess the commonest label", ACC_MAJ, GREY, f"{ACC_MAJ}%")],
     "bar_max": 100, "unit": "%",
     "foot": f"Support-ticket topics, 5 labels, 100 held-out conversations; {ACC_LO}–{ACC_HI}% across 9 training runs "
             f"(mean {ACC_MEAN}%). Demo data: synthetic, labelled by a model. Frontier accuracy: measure on our data "
             "with Marketing/benchmark/compare_frontier.py.",
     "notes": 'On a ticket-routing demo, a model trained on three hundred examples gets eighty-eight to ninety-six percent of unseen tickets right. Untrained, the same reader manages fifty-seven percent. The demo data is synthetic, so the real test is our own data, against a frontier model, with the included script.'},
    {"key": "speed", "kicker": "Speed (measured)",
     "big": f"{P50} ms", "big_sub": f"median answer, {P95} ms at the 95th percentile, on one laptop",
     "facts": [f"{RPS:.0f} answers per second per laptop: about {PER_DAY_M:.1f} million a day",
               "No network hop, no queue at a provider, no rate limits",
               "Repeated texts answer from cache in about 1 ms"],
     "foot": "Apple M5 Pro laptop, Nemotron-3-Embed-1B, end to end. A GPU server is faster. Frontier latency includes "
             "network and generation time; the comparison script measures it.",
     "notes": "It's fast: a median of thirty-three milliseconds on one laptop, with no network hop, queue or rate limit. One laptop handles about thirty-six answers a second, roughly three million a day."},
    {"key": "cost", "kicker": "Cost per million decisions",
     "title": "7× to 300× cheaper per million decisions",
     "bars": [("CLM on own hardware", CLM_COST, TEAL, f"≈ ${CLM_COST:,.0f}"),
              ("Haiku 5.5", COSTS["Haiku 5.5"], GREY, f"${COSTS['Haiku 5.5']:,.0f}"),
              ("Sonnet 5.5", COSTS["Sonnet 5.5"], GREY, f"${COSTS['Sonnet 5.5']:,.0f}"),
              ("Opus 5.5", COSTS["Opus 5.5"], ORANGE, f"${COSTS['Opus 5.5']:,.0f}")],
     "bar_max": COSTS["Opus 5.5"], "unit": "$",
     "foot": f"CLM: {HOURS_PER_M:.1f} compute-hours per million at {RPS:.0f}/s, priced at an assumed ${COMPUTE_RATE:.2f}/hour. "
             f"Frontier: Anthropic list prices (Oct 2026) at an estimated {TOK_IN} input and {TOK_OUT} output tokens per "
             "ticket. A floor: thinking tokens add to it. The Batch API halves frontier prices but adds hours of delay.",
     "notes": "And it's cheap. A million decisions take about eight hours of laptop compute, around eight dollars. At list prices, the same million tickets cost about fifty-eight dollars on the smallest frontier model, and over a thousand on the larger ones. Conservative estimates, for the frontier side."},
    {"key": "control", "kicker": "Control",
     "title": "It stays ours: data, decisions and evidence",
     "bullets": ["Texts never leave our machines; no per-request third party",
                 "Same input, same answer: training and serving are reproducible",
                 "Verified, versioned model files; a full governance pack in the repo",
                 "Commercial-use encoder from NVIDIA, built on Mistral AI; lineage documented"],
     "notes": 'It stays ours. Texts never leave our machines, the same input gives the same answer, and model files are versioned and verified, with a full governance pack in the repository.'},
    {"key": "when", "kicker": "Choosing",
     "title": "Use each where it is strongest",
     "table": [("", "CLM", "Frontier model"),
               ("Fixed labels, with examples", "Best fit", "Works, at a cost"),
               ("Cost per million decisions", "≈ $8", "$58 to $2,320"),
               ("Answer in milliseconds", "Yes", "Adds network and generation time"),
               ("Data must stay inside", "Yes", "Leaves our network"),
               ("Open questions, writing, reasoning", "No", "Best fit"),
               ("No examples, labels change daily", "No", "Best fit")],
     "foot": "Best of both: let CLM decide, and send its low-confidence cases to a frontier model or a person.",
     "notes": "Use each where it's strongest. C L M for fixed labels, high volume, and fast, private answers. A frontier model for open questions, writing and reasoning. Best of both: C L M decides, and only its uncertain cases go to a frontier model or a person."},
    {"key": "takes", "kicker": "What it takes",
     "title": "Days, not quarters",
     "steps": [("300+ labelled examples", "Exported from the system that already handles the work."),
               ("Minutes to train", "In a browser: upload, describe the labels, train, test."),
               ("One laptop or small server", "No GPU required; standalone, no external calls.")],
     "foot": "Before real decisions: human-checked labels, calibrated probabilities, and the new-model checklist.",
     "notes": 'Getting started is small: a few hundred labelled examples, minutes of training in a browser, and a laptop or small server. Before real decisions, people check the labels, the probabilities are calibrated, and the governance checklist is signed off.'},
    {"key": "ask", "kicker": "Recommendation",
     "title": "A four-week pilot on one workflow",
     "bullets": ["Pick one high-volume routing or triage workflow",
                 "Label 500 real examples; train and test CLM",
                 "Run the same test through a frontier model with the included script",
                 "Decide on measured accuracy, cost and latency, with governance sign-off"],
     "notes": 'We recommend a four-week pilot on one high-volume workflow: label five hundred real examples, test C L M and a frontier model side by side with the included script, and decide on measured accuracy, cost and latency. Thank you.'},
]

# ----------------------------------------------------------------------------- HTML rendering
CSS = f"""
*{{box-sizing:border-box}} body{{margin:0;background:{BG}}}
.s{{width:1920px;height:1080px;padding:96px 120px 80px;position:relative;overflow:hidden;color:{INK};
   font-family:-apple-system,BlinkMacSystemFont,"Helvetica Neue",Arial,sans-serif;background:{BG};display:flex;flex-direction:column}}
.s::before{{content:"";position:absolute;left:0;top:0;bottom:0;width:18px;background:{TEAL}}}
.k{{font:700 24px/1 {FF};letter-spacing:.16em;text-transform:uppercase;color:{TEAL};margin:0 0 26px}}
h1{{font:700 72px/1.08 {FF};letter-spacing:-.02em;margin:0 0 30px;max-width:1500px}}
.sub{{font:400 38px/1.35 {FF};color:{MUTED};max-width:1400px;margin:0}}
ul{{list-style:none;margin:20px 0 0;padding:0;display:grid;gap:30px}}
li{{font:400 40px/1.3 {FF};padding-left:44px;position:relative;max-width:1500px}}
li::before{{content:"";position:absolute;left:0;top:18px;width:16px;height:16px;border-radius:50%;background:{TEAL}}}
.steps{{display:grid;grid-template-columns:repeat(3,1fr);gap:40px;margin-top:40px}}
.step{{border-top:6px solid {TEAL};padding-top:28px}}
.step b{{display:block;font:700 40px/1.2 {FF};margin-bottom:16px}}
.step span{{font:400 30px/1.4 {FF};color:{MUTED}}}
.bars{{display:grid;gap:34px;margin-top:40px}}
.bar{{display:grid;grid-template-columns:560px 1fr 200px;align-items:center;gap:30px}}
.bar .l{{font:500 34px/1.2 {FF}}} .bar .v{{font:700 40px/1 {FF};text-align:right}}
.track{{height:56px;background:#f0f2f5;border-radius:6px;overflow:hidden}} .fill{{height:100%;border-radius:6px}}
.foot{{position:absolute;left:120px;right:120px;bottom:56px;font:400 22px/1.45 {FF};color:{MUTED};border-top:2px solid {LINE};padding-top:18px}}
.big{{font:800 230px/1 {FF};letter-spacing:-.04em;color:{TEAL};margin:10px 0 6px}}
.bigsub{{font:400 38px/1.3 {FF};color:{MUTED};margin-bottom:40px}}
table{{border-collapse:collapse;margin-top:26px;font:400 32px/1.3 {FF};width:1500px}}
td,th{{padding:18px 24px;border-bottom:2px solid {LINE};text-align:left}}
th{{font:700 30px {FF};color:{MUTED};text-transform:uppercase;letter-spacing:.06em}}
td.yes{{color:{TEAL};font-weight:700}} td.no{{color:{ORANGE};font-weight:700}}
.title-s{{justify-content:center}} .title-s h1{{font-size:104px;max-width:1550px}}
.brand{{position:absolute;right:120px;top:96px;font:700 30px {FF};color:{TEAL}}}
"""


def slide_html(s: dict) -> str:
    h = f'<div class="s{" title-s" if s["key"] == "title" else ""}"><div class="brand">CLM</div>'
    h += f'<p class="k">{s["kicker"]}</p>'
    if s.get("title"):
        h += f'<h1>{s["title"]}</h1>'
    if s.get("sub"):
        h += f'<p class="sub">{s["sub"]}</p>'
    if s.get("bullets"):
        h += "<ul>" + "".join(f"<li>{b}</li>" for b in s["bullets"]) + "</ul>"
    if s.get("steps"):
        h += '<div class="steps">' + "".join(f'<div class="step"><b>{a}</b><span>{b}</span></div>' for a, b in s["steps"]) + "</div>"
    if s.get("bars"):
        h += '<div class="bars">' + "".join(
            f'<div class="bar"><span class="l">{lab}</span><div class="track"><div class="fill" style="width:{max(v / s["bar_max"] * 100, 0.6):.2f}%;background:{c}"></div></div><span class="v">{txt}</span></div>'
            for lab, v, c, txt in s["bars"]) + "</div>"
    if s.get("big"):
        h += f'<div class="big">{s["big"]}</div><div class="bigsub">{s["big_sub"]}</div>'
        h += "<ul>" + "".join(f"<li>{b}</li>" for b in s["facts"]) + "</ul>"
    if s.get("table"):
        rows = s["table"]
        h += "<table><tr>" + "".join(f"<th>{c}</th>" for c in rows[0]) + "</tr>"
        for r in rows[1:]:
            cls = lambda v: "yes" if v in ("Best fit", "Yes", "≈ $8") else ("no" if v in ("No",) else "")  # noqa: E731
            h += f"<tr><td>{r[0]}</td><td class='{cls(r[1])}'>{r[1]}</td><td class='{cls(r[2])}'>{r[2]}</td></tr>"
        h += "</table>"
    if s.get("foot"):
        h += f'<div class="foot">{s["foot"]}</div>'
    return h + "</div>"


def render_pngs(out_dir: str) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 1920, "height": 1080})
        for i, s in enumerate(SLIDES):
            page.set_content(f"<style>{CSS}</style>{slide_html(s)}")
            path = os.path.join(out_dir, f"{i + 1:02d}-{s['key']}.png")
            page.screenshot(path=path)
            paths.append(path)
        page.set_content(f"<style>{CSS} .s{{page-break-after:always}}</style>" + "".join(slide_html(s) for s in SLIDES))
        page.pdf(path=os.path.join(MKT, "CLM-Executive-Briefing.pdf"), width="1920px", height="1080px",
                 print_background=True, page_ranges=f"1-{len(SLIDES)}")
        b.close()
    return paths


# ----------------------------------------------------------------------------- PowerPoint (editable)
def build_pptx(path: str) -> None:
    from pptx import Presentation
    from pptx.chart.data import CategoryChartData
    from pptx.dml.color import RGBColor
    from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION
    from pptx.util import Inches, Pt

    rgb = lambda h: RGBColor.from_string(h.lstrip("#"))  # noqa: E731
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    blank = prs.slide_layouts[6]
    FONT = "Calibri"

    def text(slide, x, y, w, h, s, size, bold=False, color=INK, upper=False):
        tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = tb.text_frame; tf.word_wrap = True
        p = tf.paragraphs[0]; r = p.add_run(); r.text = s.upper() if upper else s
        r.font.size, r.font.bold, r.font.name, r.font.color.rgb = Pt(size), bold, FONT, rgb(color)
        return tf

    for s in SLIDES:
        sl = prs.slides.add_slide(blank)
        bar = sl.shapes.add_shape(1, 0, 0, Inches(0.12), prs.slide_height)
        bar.fill.solid(); bar.fill.fore_color.rgb = rgb(TEAL); bar.line.fill.background()
        text(sl, 0.8, 0.55, 9, 0.4, s["kicker"], 13, True, TEAL, upper=True)
        text(sl, 11.6, 0.55, 1.2, 0.4, "CLM", 16, True, TEAL)
        y = 1.0
        if s.get("title"):
            text(sl, 0.8, y if s["key"] != "title" else 2.3, 11.5, 1.6, s["title"], 40 if s["key"] != "title" else 54, True)
            y = 2.25
        if s.get("sub"):
            text(sl, 0.8, 4.3, 11.5, 1.0, s["sub"], 24, False, MUTED)
        if s.get("bullets"):
            tf = text(sl, 0.8, y, 11.5, 4.2, "• " + s["bullets"][0], 24)
            for b in s["bullets"][1:]:
                p = tf.add_paragraph(); r = p.add_run(); r.text = "• " + b
                r.font.size, r.font.name, r.font.color.rgb = Pt(24), FONT, rgb(INK); p.space_before = Pt(14)
        if s.get("steps"):
            for i, (a, b) in enumerate(s["steps"]):
                x = 0.8 + i * 3.95
                line = sl.shapes.add_shape(1, Inches(x), Inches(y + 0.2), Inches(3.6), Inches(0.06))
                line.fill.solid(); line.fill.fore_color.rgb = rgb(TEAL); line.line.fill.background()
                text(sl, x, y + 0.4, 3.6, 0.8, a, 22, True)
                text(sl, x, y + 1.3, 3.6, 1.6, b, 16, False, MUTED)
        if s.get("bars"):
            cd = CategoryChartData()
            cats = [lab for lab, *_ in s["bars"]][::-1]
            cd.categories = cats
            cd.add_series("value", [v for _, v, *_ in s["bars"]][::-1])
            gf = sl.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(0.8), Inches(y), Inches(11.6), Inches(3.9), cd)
            ch = gf.chart; ch.has_legend = False
            ch.value_axis.visible = False; ch.value_axis.has_major_gridlines = False
            ch.value_axis.maximum_scale = s["bar_max"] * 1.12; ch.value_axis.minimum_scale = 0
            ch.category_axis.tick_labels.font.size = Pt(16); ch.category_axis.tick_labels.font.name = FONT
            plot = ch.plots[0]; plot.gap_width = 60; plot.has_data_labels = True
            dl = plot.data_labels; dl.font.size = Pt(16); dl.font.bold = True; dl.position = XL_LABEL_POSITION.OUTSIDE_END
            dl.number_format = '0"%"' if s["unit"] == "%" else '"$"#,##0'; dl.number_format_is_linked = False
            for pt, (_, _, c, _) in zip(plot.series[0].points, s["bars"][::-1]):
                pt.format.fill.solid(); pt.format.fill.fore_color.rgb = rgb(c)
        if s.get("big"):
            text(sl, 0.8, 1.2, 11, 1.8, s["big"], 96, True, TEAL)
            text(sl, 0.8, 2.85, 11, 0.6, s["big_sub"], 20, False, MUTED)
            tf = text(sl, 0.8, 3.7, 11.5, 2.6, "• " + s["facts"][0], 22)
            for b in s["facts"][1:]:
                p = tf.add_paragraph(); r = p.add_run(); r.text = "• " + b
                r.font.size, r.font.name, r.font.color.rgb = Pt(22), FONT, rgb(INK); p.space_before = Pt(12)
        if s.get("table"):
            rows = s["table"]
            tb = sl.shapes.add_table(len(rows), 3, Inches(0.8), Inches(y), Inches(11.6), Inches(0.5 * len(rows))).table
            tb.columns[0].width, tb.columns[1].width, tb.columns[2].width = Inches(5.2), Inches(3.2), Inches(3.2)
            for r_i, r in enumerate(rows):
                for c_i, v in enumerate(r):
                    cell = tb.cell(r_i, c_i); cell.text = v
                    run = cell.text_frame.paragraphs[0].runs[0] if cell.text_frame.paragraphs[0].runs else None
                    if run:
                        run.font.size, run.font.name = Pt(16 if r_i else 14), FONT
                        run.font.bold = r_i == 0 or v in ("Best fit", "Yes", "≈ $8", "No")
                        run.font.color.rgb = rgb(TEAL if v in ("Best fit", "Yes", "≈ $8") else ORANGE if v == "No" else INK)
                    cell.fill.solid(); cell.fill.fore_color.rgb = rgb("#f4f6f9" if r_i == 0 else "#ffffff")
        if s.get("foot"):
            text(sl, 0.8, 6.55, 11.8, 0.8, s["foot"], 11, False, MUTED)
        sl.notes_slide.notes_text_frame.text = s["notes"]
    prs.save(path)


# ----------------------------------------------------------------------------- video
def build_video(pngs: list[str], out: str, srt: str) -> float:
    work = tempfile.mkdtemp(prefix="clm-mkt-")
    try:
        scenes, t, clips = [], 0.6, []
        for i, (s, png) in enumerate(zip(SLIDES, pngs)):
            aud = os.path.join(work, f"{i:02d}.aiff")
            d = studio.tts(s["notes"], aud)
            hold = d + 1.2
            scenes.append({"start": t, "dur": d, "audio": aud, "text": s["notes"]})
            clip = os.path.join(work, f"{i:02d}.mp4")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-t", f"{hold + (0.6 if i == 0 else 0):.2f}",
                            "-i", png, "-vf", "scale=1920:1080,format=yuv420p,fps=30", "-c:v", "libx264",
                            "-preset", "slow", "-crf", "18", "-tune", "stillimage", clip], check=True)
            clips.append(clip)
            t += hold
        lst = os.path.join(work, "list.txt")
        open(lst, "w").write("".join(f"file '{c}'\n" for c in clips))
        silent = os.path.join(work, "silent.mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", silent], check=True)
        studio.write_srt(scenes, srt)
        studio.assemble(silent, scenes, out, srt, width=1920, end=t)
        return t
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main():
    pngs = render_pngs(os.path.join(MKT, "slides"))
    build_pptx(os.path.join(MKT, "CLM-Executive-Briefing.pptx"))
    secs = build_video(pngs, os.path.join(MKT, "CLM-Executive-Briefing-Video.mp4"), os.path.join(MKT, "CLM-Executive-Briefing-Video.srt"))
    print(f"slides: {len(pngs)} · video {secs / 60:.2f} min")
    if FRONTIER:
        print("frontier results included:", ", ".join(FRONTIER))


if __name__ == "__main__":
    main()
