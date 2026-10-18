#!/usr/bin/env python3
"""Take the user guide's screenshots from a running CLM (start it first with start.sh).

    .venv/bin/python "Claude Code/tools/make_guide_screenshots.py"

Writes docs/images/guide/NN-name.png at 1440x900. It trains one real model on the bundled
sample (named "guide-demo") so the results screens are genuine.
"""
from __future__ import annotations

import os
import sys

from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import studio  # noqa: E402

REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "docs", "images", "guide")
BASE = os.environ.get("CLM_URL", "http://127.0.0.1:8700")


def main():
    os.makedirs(OUT, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
        page.goto(BASE + "/")
        page.wait_for_selector("#status.ok, #status.warn", timeout=30000)
        page.evaluate(studio.OVERLAY_JS, [1, True])          # no zoom, no cursor
        # the studio hides the app's own tooltips while recording; the guide wants one shot of them
        tips = "v => { let s = document.getElementById('guide-tips'); if (!s) { s = document.createElement('style'); s.id = 'guide-tips'; document.head.appendChild(s); } s.textContent = v ? '#tip-bubble{display:block!important}' : ''; }"

        def shot(name, sel=None, wait=500):
            """Screenshot with `sel` spotlit (dimmed surroundings, gold frame, no callout)."""
            if sel:
                page.evaluate("s => window.__studio.spot(s, '', '', 8)", sel)
            page.wait_for_timeout(wait)
            page.screenshot(path=os.path.join(OUT, name))
            page.evaluate("() => window.__studio.unspot()")
            print("  ", name)

        shot("01-playground.png")
        page.evaluate(tips, True)
        page.hover('label[for="state"] + .tip-q'); page.wait_for_timeout(900)
        shot("02-tooltip.png")
        page.evaluate(tips, False)
        page.mouse.move(700, 880)
        page.select_option("#model", "typed-decisions-1b")
        page.click("#run"); page.wait_for_selector("#view-answer .ans")
        shot("03-answers.png", "#view-answer")
        page.click('.pane-res .tab[data-view="python"]')
        shot("04-python-tab.png", "#view-python")
        page.click('.pane-res .tab[data-view="answer"]')
        page.select_option("#preset", "tools"); page.wait_for_timeout(400)
        page.click("#run"); page.wait_for_timeout(1200)
        shot("05-rank.png", "#view-answer")
        page.click("#btn-settings")
        shot("06-settings.png", "#settings")
        page.click("#btn-settings")

        page.click('.vs-btn[data-view="train"]'); page.wait_for_timeout(600)
        shot("07-train-start.png", "#t-step-1")
        page.click("#t-sample"); page.wait_for_selector("#t-data .file-chip"); page.wait_for_timeout(600)
        shot("08-train-data.png", ".col-pickers")
        page.evaluate("document.querySelector('#t-step-2').scrollIntoView({block: 'start'})")
        good = page.input_value("#t-opt-refund")
        page.fill("#t-opt-refund", ""); page.type("#t-opt-refund", "Refunds")
        shot("09-train-labels.png", "#t-opt-refund")
        page.fill("#t-opt-refund", good)
        page.evaluate("document.querySelector('#t-step-3').scrollIntoView({block: 'start'})")
        page.fill("#t-name", "guide-demo")
        page.click(".adv summary"); page.wait_for_timeout(300)
        shot("10-train-settings.png", "#t-train")
        page.click("#t-go")
        page.wait_for_function("() => (typeof T !== 'undefined' && T.job && T.job.history && T.job.history.length >= 4)", timeout=300000)
        page.evaluate("document.querySelector('#t-step-3').scrollIntoView({block: 'start'})")
        shot("11-train-progress.png", "#t-progress", wait=300)
        page.wait_for_selector("#t-step-4:not(.is-locked)", timeout=600000)
        page.wait_for_timeout(800)
        page.evaluate("document.querySelector('#t-step-4').scrollIntoView({block: 'start'})")
        shot("12-train-results.png", "#t-results .cmp")
        page.click("#t-try-go"); page.wait_for_selector("#t-try-out .ans")
        page.evaluate("document.querySelector('#t-try').scrollIntoView({block: 'start'})")
        shot("13-train-try.png", "#t-try-out")
        page.evaluate("document.querySelector('#t-export').scrollIntoView({block: 'center'})")
        page.fill("#t-export-dest", "exports/")
        shot("16-train-export.png", "#t-export")
        page.evaluate("document.querySelector('#t-models').scrollIntoView({block: 'end'})")
        shot("14-models-list.png", "#t-models")
        page.click("#t-open"); page.wait_for_timeout(800)
        page.click("#run"); page.wait_for_selector("#view-answer .ans")
        shot("15-playground-trained.png", "#model")
        browser.close()


if __name__ == "__main__":
    main()
