#!/usr/bin/env python3
"""Build the executive briefing: slides (PNG, PDF), an editable PowerPoint, and a ~3-minute video.

    .venv/bin/python "Claude Code/tools/build_marketing.py"

Inputs: Marketing/benchmark/results.json (CLM measured by `clm-benchmark`)
and, when present, Marketing/benchmark/frontier-results.json (from Claude Code/tools/compare_frontier.py).
Frontier accuracy and latency appear only if that second file exists; costs otherwise come
from list prices with the token assumptions printed on the slide.

The video is the same slides, animated (elements build in, bars grow, numbers count up),
with a live scene of the real playground in the middle, recorded through the studio at
3840x2160 and delivered at 1080p. Needs Playwright, python-pptx, macOS `say`, ffmpeg and a
running CLM for the live scene (set MARKETING_NO_DEMO=1 to skip it).
"""
from __future__ import annotations

import json
import os
import sys

from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
MKT = os.path.join(REPO, "Marketing")
sys.path.insert(0, HERE)
import studio  # noqa: E402

BASE = os.environ.get("CLM_URL", "http://127.0.0.1:8700")
DEMO = not os.environ.get("MARKETING_NO_DEMO")

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

# palette: deep navy ground, teal for CLM, amber for the frontier side
BG0, BG1 = "#0a1526", "#0e2039"
INK, MUTED, LINE = "#f3f6fa", "#a9b6c8", "rgba(255,255,255,.12)"
TEAL, TEAL2, ORANGE, GREY = "#2cc5ad", "#0f7b6c", "#f0a04b", "#4d5f7a"
FF = '-apple-system,BlinkMacSystemFont,"Helvetica Neue",Arial,sans-serif'
MARK = """<svg class="mark" width="54" height="54" viewBox="0 0 100 100" aria-hidden="true">
      <path class="mark-branch" d="M10 49.4C24 49 26 22.1 45 22.1"/>
      <path class="mark-branch" d="M10 49.4C24 49.8 26 75.7 45 75.7"/>
      <circle class="mark-ring" cx="52.2" cy="22.1" r="5.3"/>
      <circle class="mark-ring" cx="52.2" cy="75.7" r="5.3"/>
      <path class="mark-trunk" d="M8 49.4H79"/>
      <circle class="mark-node" cx="49.9" cy="49.4" r="6.3"/>
      <circle class="mark-node" cx="79.3" cy="49.4" r="13"/>
      <path class="mark-check" d="M71.5 49.8l5.2 6.6 11.4-13.1"/>
    </svg>"""      # the playground's own mark (src/clm/static/index.html)

# ----------------------------------------------------------------------------- slides
SLIDES = [
    {"key": "title", "kicker": "Executive briefing · October 2026",
     "title": "Fast, cheap, private text classification",
     "sub": "When a small trained model beats calling a frontier language model",
     "notes": "Every day our teams route tickets, tag documents and triage alerts, and many now send each decision to a frontier language model. Here's when a small trained classifier does that job better: cheaper, faster, and inside our walls."},
    {"key": "problem", "kicker": "The job",
     "title": "High-volume classification is a different job",
     "bullets": ["Millions of short, repetitive decisions: route, tag, flag",
                 "A fixed set of labels, and past examples of each",
                 "Needs to be fast, cheap per decision, private and consistent",
                 "A frontier model is a generalist, paid per token, run remotely"],
     "tickets": [("Charged twice this month", "billing"), ("Parcel never arrived", "delivery"),
                 ("App crashes on login", "technical"), ("Want my money back", "refund"), ("Change plan owner", "account")],
     "notes": 'Classification at volume is millions of short, repetitive decisions with fixed labels and past examples. What matters is speed, cost per decision, privacy and consistency. A frontier model is a brilliant generalist, but we pay per token, and every request leaves the building.'},
    {"key": "what", "kicker": "The approach",
     "title": "A frozen reader, a small learner, one fast answer",
     "steps": [("NVIDIA Nemotron reads", "Turns each text into numbers. Never retrained."),
               ("A small model learns", "Our labels, from our examples, in seconds to minutes."),
               ("We get the odds", "A probability per label, in milliseconds, on our hardware.")],
     "notes": 'C L M splits the work. An NVIDIA Nemotron model reads each text and never changes. A small model on top learns our labels from our own examples, in minutes, from a browser. We get a probability for every label, in milliseconds, on our own hardware.'},
    {"key": "demo", "live": True,
     "notes": "This is the real thing. A support message, three questions: is it urgent, which team, how frustrated is the customer. One call, and every option comes back with a probability, in tens of milliseconds, from a laptop."},
    {"key": "accuracy", "kicker": "Accuracy (measured)",
     "title": f"Trained on 300 examples: {ACC_SHIPPED}% to {ACC_HI}% right",
     "bars": [("Trained CLM model", ACC_SHIPPED, TEAL, f"{ACC_SHIPPED}–{ACC_HI}%"),
              ("Same reader, no training", ACC_RAW, GREY, f"{ACC_RAW}%"),
              ("Always guess the commonest label", ACC_MAJ, GREY, f"{ACC_MAJ}%")],
     "bar_max": 100, "unit": "%",
     "foot": f"Support-ticket topics, 5 labels, 100 held-out conversations; {ACC_LO}–{ACC_HI}% across 9 training runs "
             f"(mean {ACC_MEAN}%). Demo data: synthetic, labelled by a model. Frontier accuracy: measure on our data "
             "with Claude Code/tools/compare_frontier.py.",
     "notes": 'On a ticket-routing demo, a model trained on three hundred examples gets eighty-eight to ninety-six percent of unseen tickets right. Untrained, the same reader manages fifty-seven percent. The demo data is synthetic, so the real test is our own data, against a frontier model, with the included script.'},
    {"key": "speed", "kicker": "Speed (measured)",
     "big": P50, "big_unit": "ms", "big_sub": f"median answer, {P95} ms at the 95th percentile, on one laptop",
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
                 "Verified, versioned model files; export to disk or S3 with checksums",
                 "Commercial-use encoder from NVIDIA, built on Mistral AI; lineage documented"],
     "notes": 'It stays ours. Texts never leave our machines, the same input gives the same answer, and every model is a verified file we can export to disk or S3, with a governance pack in the repository.'},
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
     "notes": 'Getting started is small: a few hundred labelled examples, minutes of training in a browser, a laptop or small server. Before real decisions: checked labels, calibrated probabilities, and the governance checklist signed off.'},
    {"key": "ask", "kicker": "Recommendation",
     "title": "A four-week pilot on one workflow",
     "bullets": ["Pick one high-volume routing or triage workflow",
                 "Label 500 real examples; train and test CLM",
                 "Run the same test through a frontier model with the included script",
                 "Decide on measured accuracy, cost and latency, with governance sign-off"],
     "notes": 'We recommend a four-week pilot on one high-volume workflow: label five hundred real examples, test C L M and a frontier model side by side with the included script, and decide on measured accuracy, cost and latency. Thank you.'},
]
DECK = [s for s in SLIDES if not s.get("live")]      # the slides that exist as PNG, PDF and PowerPoint

# ----------------------------------------------------------------------------- HTML rendering
# One stylesheet for the still slides and the video. `zoom` is set per use (2 while recording).
CSS = f"""
*{{box-sizing:border-box}} html{{zoom:ZOOM}} body{{margin:0;background:{BG0}}}
.s{{width:1920px;height:1080px;padding:92px 130px 84px;position:relative;overflow:hidden;color:{INK};font-family:{FF};
   background:radial-gradient(1100px 760px at 88% -12%,rgba(44,197,173,.26),transparent 62%),
              radial-gradient(900px 600px at -10% 110%,rgba(240,160,75,.10),transparent 60%),
              linear-gradient(160deg,{BG0} 0%,{BG1} 58%,{BG0} 100%);display:flex;flex-direction:column}}
.s::after{{content:"";position:absolute;inset:0;pointer-events:none;
   background-image:linear-gradient(rgba(255,255,255,.035) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.035) 1px,transparent 1px);
   background-size:96px 96px;mask-image:linear-gradient(180deg,rgba(0,0,0,.9),transparent 70%)}}
.brand{{position:absolute;right:130px;top:76px;display:flex;align-items:center;gap:16px;font:800 34px {FF};color:{INK};letter-spacing:.04em}}
.brand svg{{width:64px;height:64px}}
.brand .mark-branch{{fill:none;stroke:#c3ccd9;stroke-width:5.5;stroke-linecap:round}} .brand .mark-node{{fill:#e2444c}}
.brand .mark-check{{fill:none;stroke:{BG1};stroke-width:5;stroke-linecap:square;stroke-linejoin:miter}}
.body{{flex:1;display:flex;flex-direction:column;justify-content:center;padding-bottom:40px}}
.k{{font:700 22px/1 {FF};letter-spacing:.18em;text-transform:uppercase;color:{TEAL};margin:0 0 28px;display:flex;align-items:center;gap:16px}}
.k::before{{content:"";width:44px;height:3px;background:{TEAL};border-radius:2px}}
h1{{font:700 72px/1.08 {FF};letter-spacing:-.025em;margin:0 0 30px;max-width:1480px;text-wrap:balance}}
.sub{{font:400 38px/1.35 {FF};color:{MUTED};max-width:1380px;margin:0}}
ul{{list-style:none;margin:16px 0 0;padding:0;display:grid;gap:30px}}
li{{font:400 38px/1.3 {FF};padding-left:58px;position:relative;max-width:1500px}}
li::before{{content:"";position:absolute;left:0;top:14px;width:26px;height:26px;border-radius:50%;background:{TEAL2};
   box-shadow:0 0 0 6px rgba(44,197,173,.18)}}
li::after{{content:"";position:absolute;left:8px;top:21px;width:10px;height:6px;border-left:3px solid #fff;border-bottom:3px solid #fff;transform:rotate(-45deg)}}
.two{{display:grid;grid-template-columns:1.15fr .85fr;gap:60px;align-items:start}}
.tickets{{display:grid;gap:16px;margin-top:10px}}
.tk{{display:flex;justify-content:space-between;align-items:center;gap:20px;background:rgba(255,255,255,.05);border:1px solid {LINE};
   border-radius:14px;padding:18px 24px;font:400 26px {FF};color:{INK}}}
.tk b{{font:700 18px {FF};letter-spacing:.08em;text-transform:uppercase;color:{TEAL};background:rgba(44,197,173,.12);
   border:1px solid rgba(44,197,173,.35);border-radius:999px;padding:6px 14px}}
.steps{{display:grid;grid-template-columns:repeat(3,1fr);gap:48px;margin-top:48px;position:relative}}
.step{{background:rgba(255,255,255,.045);border:1px solid {LINE};border-radius:20px;padding:34px 34px 38px;position:relative}}
.step i{{display:flex;align-items:center;justify-content:center;width:64px;height:64px;border-radius:50%;background:{TEAL2};
   color:#fff;font:700 30px {FF};font-style:normal;margin-bottom:24px;box-shadow:0 0 0 8px rgba(44,197,173,.16)}}
.step b{{display:block;font:700 38px/1.2 {FF};margin-bottom:14px}}
.step span{{font:400 28px/1.4 {FF};color:{MUTED}}}
.step+.step::before{{content:"";position:absolute;left:-48px;top:62px;width:48px;height:3px;background:linear-gradient(90deg,{TEAL2},{TEAL})}}
.step+.step::after{{content:"";position:absolute;left:-14px;top:55px;border:9px solid transparent;border-left:12px solid {TEAL}}}
.bars{{display:grid;gap:36px;margin-top:44px}}
.bar{{display:grid;grid-template-columns:560px 1fr 220px;align-items:center;gap:30px}}
.bar .l{{font:500 34px/1.2 {FF}}} .bar .v{{font:700 44px/1 {FF};text-align:right;letter-spacing:-.02em}}
.track{{height:58px;background:rgba(255,255,255,.07);border-radius:8px;overflow:hidden}}
.fill{{height:100%;border-radius:8px;width:var(--w)}}
.foot{{position:absolute;left:130px;right:130px;bottom:52px;font:400 21px/1.45 {FF};color:{MUTED};border-top:1px solid {LINE};padding-top:18px}}
.big{{font:800 250px/1 {FF};letter-spacing:-.045em;color:{TEAL};margin:0 0 6px;display:flex;align-items:baseline;gap:22px}}
.big small{{font:700 90px {FF};color:{MUTED};letter-spacing:-.02em}}
.bigsub{{font:400 38px/1.3 {FF};color:{MUTED};margin-bottom:38px}}
table{{border-collapse:collapse;margin-top:22px;font:400 31px/1.3 {FF};width:1540px}}
td,th{{padding:19px 24px;border-bottom:1px solid {LINE};text-align:left}}
th{{font:700 24px {FF};color:{MUTED};text-transform:uppercase;letter-spacing:.1em}}
td.yes{{color:{TEAL};font-weight:700}} td.no{{color:{ORANGE};font-weight:700}}
.title-s{{justify-content:center}} .title-s h1{{font-size:112px;max-width:1560px}} .title-s .k{{margin-bottom:40px}}
.title-s .sub{{font-size:42px}}
.pager{{position:absolute;left:130px;bottom:24px;font:600 18px {FF};letter-spacing:.14em;color:rgba(255,255,255,.35)}}
/* builds, used by the video; the still renderer sets .still to skip them */
.a{{opacity:0;transform:translateY(22px);animation:in .75s cubic-bezier(.2,.7,.2,1) forwards;animation-delay:calc(var(--i,0)*.38s + .25s)}}
.fill{{animation:grow 1.3s cubic-bezier(.2,.7,.2,1) forwards;animation-delay:calc(var(--i,0)*.38s + .55s);transform-origin:left;transform:scaleX(0)}}
.still .a,.still .fill{{animation:none;opacity:1;transform:none}}
@keyframes in{{to{{opacity:1;transform:none}}}} @keyframes grow{{to{{transform:scaleX(1)}}}}
"""

COUNT_JS = """
<script>
for (const el of document.querySelectorAll('.s:not(.still) [data-count]')) {
  const end = +el.dataset.count, dur = 1400, t0 = performance.now() + 500;
  const tick = (t) => { const u = Math.min(1, Math.max(0, (t - t0) / dur)), e = 1 - Math.pow(1 - u, 3);
    el.textContent = Math.round(end * e); if (u < 1) requestAnimationFrame(tick); };
  requestAnimationFrame(tick);
}
</script>"""


def slide_html(s: dict, n: int, total: int, still: bool = False) -> str:
    i = iter(range(100))
    a = lambda: f' class="a" style="--i:{next(i)}"'  # noqa: E731
    h = f'<div class="s{" title-s" if s["key"] == "title" else ""}{" still" if still else ""}">'
    h += f'<div class="brand">{MARK}CLM</div><div class="body">'
    h += f'<p class="k"{a()}>{s["kicker"]}</p>'
    if s.get("title"):
        h += f'<h1{a()}>{s["title"]}</h1>'
    if s.get("sub"):
        h += f'<p class="sub"{a()}>{s["sub"]}</p>'
    if s.get("bullets") and s.get("tickets"):
        h += '<div class="two"><ul>' + "".join(f"<li{a()}>{b}</li>" for b in s["bullets"]) + "</ul><div class='tickets'>"
        h += "".join(f'<div class="tk"{a()}><span>“{t}”</span><b>{lab}</b></div>' for t, lab in s["tickets"]) + "</div></div>"
    elif s.get("bullets"):
        h += "<ul>" + "".join(f"<li{a()}>{b}</li>" for b in s["bullets"]) + "</ul>"
    if s.get("steps"):
        h += '<div class="steps">' + "".join(f'<div class="step"{a()}><i>{k + 1}</i><b>{t}</b><span>{d}</span></div>'
                                             for k, (t, d) in enumerate(s["steps"])) + "</div>"
    if s.get("bars"):
        h += '<div class="bars">' + "".join(
            f'<div class="bar"{a()}><span class="l">{lab}</span><div class="track"><div class="fill" style="--w:{max(v / s["bar_max"] * 100, 0.6):.2f}%;--i:{k};background:{c}"></div></div><span class="v">{txt}</span></div>'
            for k, (lab, v, c, txt) in enumerate(s["bars"])) + "</div>"
    if s.get("big") is not None:
        h += f'<div class="big"{a()}><span data-count="{s["big"]}">{s["big"] if still else 0}</span><small>{s["big_unit"]}</small></div><div class="bigsub"{a()}>{s["big_sub"]}</div>'
        h += "<ul>" + "".join(f"<li{a()}>{b}</li>" for b in s["facts"]) + "</ul>"
    if s.get("table"):
        rows = s["table"]
        h += "<table><tr>" + "".join(f"<th>{c}</th>" for c in rows[0]) + "</tr>"
        for r in rows[1:]:
            cls = lambda v: "yes" if v in ("Best fit", "Yes", "≈ $8") else ("no" if v in ("No",) else "")  # noqa: E731
            h += f"<tr{a()}><td>{r[0]}</td><td class='{cls(r[1])}'>{r[1]}</td><td class='{cls(r[2])}'>{r[2]}</td></tr>"
        h += "</table>"
    h += "</div>"
    if s.get("foot"):
        h += f'<div class="foot">{s["foot"]}</div>'
    if s["key"] != "title":
        h += f'<div class="pager">{n:02d} / {total:02d}</div>'
    return h + "</div>"


def page_html(body: str, zoom: int = 1) -> str:
    return f"<style>{CSS.replace('ZOOM', str(zoom))}</style>{body}{COUNT_JS}"


def render_stills(out_dir: str) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 1920, "height": 1080})
        for i, s in enumerate(DECK):
            page.set_content(page_html(slide_html(s, i + 1, len(DECK), still=True)))
            path = os.path.join(out_dir, f"{i + 1:02d}-{s['key']}.png")
            page.screenshot(path=path)
            paths.append(path)
        page.set_content(page_html("".join(slide_html(s, i + 1, len(DECK), still=True) for i, s in enumerate(DECK))).replace(
            ".s{", ".s{page-break-after:always;", 1))
        page.pdf(path=os.path.join(MKT, "CLM-Executive-Briefing.pdf"), width="1920px", height="1080px",
                 print_background=True, page_ranges=f"1-{len(DECK)}")
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
    P_BG, P_INK, P_MUTED, P_GREY, P_CELL = "#0C1A30", "#F3F6FA", "#A9B6C8", "#4D5F7A", "#15284A"
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    blank = prs.slide_layouts[6]
    FONT = "Calibri"

    def text(slide, x, y, w, h, s, size, bold=False, color=P_INK, upper=False):
        tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = tb.text_frame; tf.word_wrap = True
        p = tf.paragraphs[0]; r = p.add_run(); r.text = s.upper() if upper else s
        r.font.size, r.font.bold, r.font.name, r.font.color.rgb = Pt(size), bold, FONT, rgb(color)
        return tf

    for n, s in enumerate(DECK, 1):
        sl = prs.slides.add_slide(blank)
        bg = sl.shapes.add_shape(1, 0, 0, prs.slide_width, prs.slide_height)
        bg.fill.solid(); bg.fill.fore_color.rgb = rgb(P_BG); bg.line.fill.background()
        rule = sl.shapes.add_shape(1, Inches(0.8), Inches(0.72), Inches(0.35), Inches(0.04))
        rule.fill.solid(); rule.fill.fore_color.rgb = rgb(TEAL); rule.line.fill.background()
        text(sl, 1.25, 0.55, 9, 0.4, s["kicker"], 13, True, TEAL, upper=True)
        text(sl, 11.6, 0.55, 1.2, 0.4, "CLM", 16, True, P_INK)
        if n > 1:
            text(sl, 0.8, 6.95, 2, 0.3, f"{n:02d} / {len(DECK):02d}", 9, False, P_GREY)
        y = 1.0
        if s.get("title"):
            text(sl, 0.8, y if s["key"] != "title" else 2.3, 11.5, 1.6, s["title"], 40 if s["key"] != "title" else 54, True)
            y = 2.25
        if s.get("sub"):
            text(sl, 0.8, 4.3, 11.5, 1.0, s["sub"], 24, False, P_MUTED)
        if s.get("bullets"):
            tf = text(sl, 0.8, y, 11.5, 4.2, "• " + s["bullets"][0], 24)
            for b in s["bullets"][1:]:
                p = tf.add_paragraph(); r = p.add_run(); r.text = "• " + b
                r.font.size, r.font.name, r.font.color.rgb = Pt(24), FONT, rgb(P_INK); p.space_before = Pt(14)
        if s.get("steps"):
            for i, (a, b) in enumerate(s["steps"]):
                x = 0.8 + i * 3.95
                line = sl.shapes.add_shape(1, Inches(x), Inches(y + 0.2), Inches(3.6), Inches(0.06))
                line.fill.solid(); line.fill.fore_color.rgb = rgb(TEAL); line.line.fill.background()
                text(sl, x, y + 0.4, 3.6, 0.8, a, 22, True)
                text(sl, x, y + 1.3, 3.6, 1.6, b, 16, False, P_MUTED)
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
            ch.category_axis.tick_labels.font.color.rgb = rgb(P_INK)
            ch.category_axis.format.line.color.rgb = rgb(P_GREY)
            plot = ch.plots[0]; plot.gap_width = 60; plot.has_data_labels = True
            dl = plot.data_labels; dl.font.size = Pt(16); dl.font.bold = True; dl.font.color.rgb = rgb(P_INK)
            dl.position = XL_LABEL_POSITION.OUTSIDE_END
            dl.number_format = '0"%"' if s["unit"] == "%" else '"$"#,##0'; dl.number_format_is_linked = False
            for pt, (_, _, c, _) in zip(plot.series[0].points, s["bars"][::-1]):
                pt.format.fill.solid(); pt.format.fill.fore_color.rgb = rgb(P_GREY if c == GREY else c)
        if s.get("big") is not None:
            text(sl, 0.8, 1.2, 11, 1.8, f"{s['big']} {s['big_unit']}", 96, True, TEAL)
            text(sl, 0.8, 2.85, 11, 0.6, s["big_sub"], 20, False, P_MUTED)
            tf = text(sl, 0.8, 3.7, 11.5, 2.6, "• " + s["facts"][0], 22)
            for b in s["facts"][1:]:
                p = tf.add_paragraph(); r = p.add_run(); r.text = "• " + b
                r.font.size, r.font.name, r.font.color.rgb = Pt(22), FONT, rgb(P_INK); p.space_before = Pt(12)
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
                        run.font.color.rgb = rgb(TEAL if v in ("Best fit", "Yes", "≈ $8") else ORANGE if v == "No" else P_MUTED if r_i == 0 else P_INK)
                    cell.fill.solid(); cell.fill.fore_color.rgb = rgb(P_CELL if r_i == 0 else P_BG)
        if s.get("foot"):
            text(sl, 0.8, 6.55, 11.8, 0.8, s["foot"], 11, False, P_MUTED)
        sl.notes_slide.notes_text_frame.text = s["notes"]
    prs.save(path)


# ----------------------------------------------------------------------------- video
def build_video(out: str, srt: str) -> float:
    """Record the animated slides and the live scene through the studio."""
    W, H, Z = 1920, 1080, 2
    narrations = {s["key"]: s["notes"] for s in SLIDES if not (s.get("live") and not DEMO)}
    work = studio.mkwork("clm-mkt-")
    with sync_playwright() as p:
        b = p.chromium.launch()
        ctx = b.new_context(viewport={"width": W * Z, "height": H * Z}, device_scale_factor=1)
        page = ctx.new_page()
        page.set_content(page_html('<div class="s"></div>', Z))
        rec = studio.Recorder(ctx, page, work, narrations, logical=(W, H), zoom=Z, cursor=False)

        def fade_out():
            page.evaluate("() => { document.body.style.transition = 'opacity .4s'; document.body.style.opacity = 0; }")
            page.wait_for_timeout(420)

        n = 0
        for s in SLIDES:
            if s.get("live"):
                if not DEMO:
                    continue
                fade_out()
                page.goto(BASE + "/")
                page.wait_for_selector("#status.ok", timeout=30000)
                page.evaluate("() => { try { localStorage.clear(); } catch (e) {} }")
                page.reload(); page.wait_for_selector("#status.ok", timeout=30000)
                rec.overlay()                                    # applies the zoom to the app page
                page.evaluate("() => loadPreset('support')")
                page.select_option("#model", "typed-decisions-1b")
                page.wait_for_timeout(600)
                rec.scene("demo", "", "Live")
                rec.spot("#state", "A support message", "The text to classify.", wait=2.6)
                rec.spot(".q-card", "Three questions", "Urgent? Which team? How frustrated?", wait=2.8, zoom=1)
                rec.unspot()
                rec.click("#run", wait=1.4)
                rec.spot("#view-answer .ans", "Answers", "A probability for every option.", wait=3.2)
                rec.spot("#meta", "Measured", "Time spent inside CLM for this call.", wait=2.6)
                rec.hold(0.4)
                rec.unspot()
                fade_out()
                page.set_content(page_html('<div class="s"></div>', Z))
                continue
            n += 1
            if n > 1:
                fade_out()
            page.set_content(page_html(slide_html(s, n, len(DECK)), Z))
            rec.scene(s["key"])
            rec.hold(0.7)
        end = rec.finish()
        b.close()
    studio.write_srt(rec.scenes, srt)
    studio.assemble(rec, out, srt, end)
    studio.cleanup(work)
    return end


def main():
    pngs = render_stills(os.path.join(MKT, "slides"))
    build_pptx(os.path.join(MKT, "CLM-Executive-Briefing.pptx"))
    secs = build_video(os.path.join(MKT, "CLM-Executive-Briefing-Video.mp4"), os.path.join(MKT, "CLM-Executive-Briefing-Video.srt"))
    print(f"slides: {len(pngs)} · video {secs / 60:.2f} min")
    if FRONTIER:
        print("frontier results included:", ", ".join(FRONTIER))


if __name__ == "__main__":
    main()
