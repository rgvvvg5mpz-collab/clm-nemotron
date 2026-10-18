"""Small video studio: narrate with macOS speech, drive the real app with Playwright, assemble with ffmpeg.

Used by the Crew Training video, the Marketing video and the user-guide screenshots.

Design: narration for each scene is synthesised first, so its length is known. The browser
is recorded continuously; each scene's actions run, then the scene waits until its
narration has finished. The soundtrack places each narration clip at its scene's start
time. Captions are drawn inside the page (this ffmpeg has no text filters) and also
written as an .srt file plus a soft subtitle track.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time

VOICE = os.environ.get("STUDIO_VOICE", "Ava (Premium)")
RATE = os.environ.get("STUDIO_RATE", "182")          # words per minute for `say`


# ----------------------------------------------------------------------------- audio
def tts(text: str, path: str) -> float:
    """Speak `text` into `path` (.aiff); returns its duration in seconds."""
    clean = re.sub(r"\s+", " ", text).strip()
    subprocess.run(["say", "-v", VOICE, "-r", RATE, "-o", path, clean], check=True)
    return duration(path)


def duration(path: str) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", path],
                         capture_output=True, text=True, check=True).stdout
    return float(json.loads(out)["format"]["duration"])


# ----------------------------------------------------------------------------- overlay
OVERLAY_JS = r"""
(() => {
  if (window.__studio) return;
  const css = `
  #studio-cap{position:fixed;left:50%;bottom:26px;transform:translateX(-50%);z-index:2147483646;max-width:min(1100px,90vw);
    background:rgba(18,18,22,.86);color:#fff;font:500 21px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
    padding:12px 22px;border-radius:12px;box-shadow:0 8px 30px rgba(0,0,0,.35);text-align:center;transition:opacity .25s}
  #studio-step{position:fixed;right:24px;top:62px;z-index:2147483646;background:#b1040e;color:#fff;border-radius:999px;
    font:700 15px -apple-system,BlinkMacSystemFont,sans-serif;padding:7px 15px;box-shadow:0 4px 14px rgba(0,0,0,.25);transition:opacity .25s}
  #studio-cursor{position:fixed;z-index:2147483647;width:22px;height:22px;margin:-3px 0 0 -3px;pointer-events:none;
    transition:left .55s cubic-bezier(.3,.7,.3,1),top .55s cubic-bezier(.3,.7,.3,1)}
  #studio-cursor.click::after{content:"";position:absolute;left:-12px;top:-12px;width:44px;height:44px;border-radius:50%;
    border:3px solid rgba(177,4,14,.75);animation:studio-ping .5s ease-out forwards}
  @keyframes studio-ping{from{transform:scale(.3);opacity:1}to{transform:scale(1.2);opacity:0}}
  .studio-ring{position:fixed;z-index:2147483645;border:3px solid #e9a200;border-radius:10px;pointer-events:none;
    box-shadow:0 0 0 6px rgba(233,162,0,.22);transition:all .35s}
  #studio-card{position:fixed;inset:0;z-index:2147483640;display:flex;align-items:center;justify-content:center;
    background:#0f1115;color:#f3f1ef;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;transition:opacity .35s}
  #studio-card .inner{width:min(1180px,88vw)}
  `;
  const st = document.createElement('style'); st.textContent = css; document.head.appendChild(st);
  const cap = document.createElement('div'); cap.id = 'studio-cap'; cap.style.opacity = 0; document.body.appendChild(cap);
  const step = document.createElement('div'); step.id = 'studio-step'; step.style.opacity = 0; document.body.appendChild(step);
  const cur = document.createElement('div'); cur.id = 'studio-cursor';
  cur.innerHTML = '<svg viewBox="0 0 24 24" width="22" height="22"><path d="M3 2l7 19 2.6-7.4L20 11z" fill="#111" stroke="#fff" stroke-width="1.6" stroke-linejoin="round"/></svg>';
  cur.style.left = '50vw'; cur.style.top = '60vh'; document.body.appendChild(cur);
  window.__studio = {
    caption(t) { cap.textContent = t || ''; cap.style.opacity = t ? 1 : 0; },
    step(t) { step.textContent = t || ''; step.style.opacity = t ? 1 : 0; },
    move(x, y) { cur.style.left = x + 'px'; cur.style.top = y + 'px'; },
    click() { cur.classList.remove('click'); void cur.offsetWidth; cur.classList.add('click'); },
    ring(sel) {
      document.querySelectorAll('.studio-ring').forEach(r => r.remove());
      if (!sel) return;
      const el = typeof sel === 'string' ? document.querySelector(sel) : sel; if (!el) return;
      const r = el.getBoundingClientRect(); const d = document.createElement('div'); d.className = 'studio-ring';
      Object.assign(d.style, {left: (r.left - 6) + 'px', top: (r.top - 6) + 'px', width: (r.width + 12) + 'px', height: (r.height + 12) + 'px'});
      document.body.appendChild(d);
    },
    card(html) {
      let c = document.getElementById('studio-card');
      if (!html) { cur.style.opacity = 1; if (c) { c.style.opacity = 0; setTimeout(() => c.remove(), 400); } return; }
      this.ring(null); this.step(''); this.caption(''); cur.style.opacity = 0;    // a card is a clean slide
      if (!c) { c = document.createElement('div'); c.id = 'studio-card'; c.style.opacity = 0; document.body.appendChild(c); }
      c.innerHTML = '<div class="inner">' + html + '</div>'; requestAnimationFrame(() => c.style.opacity = 1);
    },
  };
})();
"""


class Recorder:
    """One continuous recording of a Playwright page, with narrated scenes."""

    def __init__(self, page, workdir: str):
        self.page, self.workdir = page, workdir
        self.t0 = time.monotonic()
        self.scenes: list[dict] = []
        self.audio_dir = os.path.join(workdir, "audio")
        os.makedirs(self.audio_dir, exist_ok=True)

    def now(self) -> float:
        return time.monotonic() - self.t0

    def overlay(self):
        self.page.evaluate(OVERLAY_JS)

    def js(self, expr: str, *args):
        self.overlay()
        return self.page.evaluate(expr, *args) if args else self.page.evaluate(expr)

    # --- scene timing
    def scene(self, key: str, narration: str, caption: str = "", step: str | None = None):
        """Start a scene: synthesise its narration and note when it begins."""
        path = os.path.join(self.audio_dir, f"{len(self.scenes):02d}-{key}.aiff")
        dur = tts(narration, path)
        s = {"key": key, "start": self.now(), "audio": path, "dur": dur, "text": narration, "caption": caption}
        self.scenes.append(s)
        self.overlay()
        self.page.evaluate("t => window.__studio.caption(t)", caption)
        if step is not None:
            self.page.evaluate("t => window.__studio.step(t)", step)
        return s

    def hold(self, extra: float = 0.6):
        """Wait until the current scene's narration has finished (plus a short breath)."""
        s = self.scenes[-1]
        left = s["start"] + s["dur"] + extra - self.now()
        if left > 0:
            self.page.wait_for_timeout(int(left * 1000))
        s["end"] = self.now()

    # --- actions with a visible cursor
    def point(self, selector: str, ring: bool = True, wait: float = 0.7):
        self.overlay()
        el = self.page.locator(selector).first
        el.scroll_into_view_if_needed()
        box = el.bounding_box()
        if box:
            self.page.evaluate("([x, y]) => window.__studio.move(x, y)", [box["x"] + box["width"] / 2, box["y"] + box["height"] / 2])
        if ring:
            self.page.evaluate("s => window.__studio.ring(s)", selector)
        self.page.wait_for_timeout(int(wait * 1000))
        return el

    def click(self, selector: str, wait: float = 0.6):
        el = self.point(selector)
        self.page.evaluate("() => window.__studio.click()")
        el.click()
        self.page.evaluate("() => window.__studio.ring(null)")
        self.page.wait_for_timeout(int(wait * 1000))

    def type(self, selector: str, text: str, delay: int = 28, clear: bool = True):
        el = self.point(selector)
        el.click()
        if clear:
            el.fill("")
        el.type(text, delay=delay)
        self.page.evaluate("() => window.__studio.ring(null)")

    def card(self, html: str | None):
        self.overlay()
        self.page.evaluate("h => window.__studio.card(h)", html)

    def scroll_to(self, selector: str, block: str = "start"):
        self.page.evaluate("([s, b]) => document.querySelector(s).scrollIntoView({behavior: 'smooth', block: b})", [selector, block])
        self.page.wait_for_timeout(700)


# ----------------------------------------------------------------------------- assembly
def srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# spellings written for the speech engine, shown normally in subtitles
SPOKEN = {"C L M": "CLM", "C S V": "CSV", "I D": "ID", "localhost, port eighty-seven hundred": "localhost:8700"}


def display(text: str) -> str:
    for spoken, shown in SPOKEN.items():
        text = text.replace(spoken, shown)
    return text


def write_srt(scenes: list[dict], path: str, offset: float = 0.0) -> None:
    """One cue per sentence, spread over the scene's narration time by sentence length."""
    cues, n = [], 1
    for s in scenes:
        sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", s["text"])) if x.strip()]
        total = sum(len(x) for x in sentences) or 1
        t = s["start"] + offset
        for x in sentences:
            d = s["dur"] * len(x) / total
            cues.append(f"{n}\n{srt_time(t)} --> {srt_time(t + d)}\n{display(x)}\n")
            t += d; n += 1
    open(path, "w", encoding="utf-8").write("\n".join(cues))


def assemble(video_in: str, scenes: list[dict], out_mp4: str, srt: str, width: int = 1280,
             offset: float = 0.0, end: float | None = None) -> None:
    """Mux the recording with each narration placed at its scene start; add soft subtitles."""
    tmp = tempfile.mkdtemp()
    try:
        inputs, filters = ["-i", video_in], []
        for i, s in enumerate(scenes):
            inputs += ["-i", s["audio"]]
            ms = max(0, int((s["start"] + offset) * 1000))
            filters.append(f"[{i + 1}:a]aresample=48000,adelay={ms}|{ms}[a{i}]")
        mix = "".join(f"[a{i}]" for i in range(len(scenes)))
        filters.append(f"{mix}amix=inputs={len(scenes)}:normalize=0,apad[aout]")
        trim = ["-t", f"{end:.2f}"] if end else ["-shortest"]
        cmd = ["ffmpeg", "-y", "-loglevel", "error", *inputs, "-i", srt,
               "-filter_complex", ";".join(filters) + f";[0:v]scale={width}:-2:flags=lanczos,format=yuv420p[vout]",
               "-map", "[vout]", "-map", "[aout]", "-map", f"{len(scenes) + 1}:s",
               "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-c:a", "aac", "-b:a", "160k",
               "-c:s", "mov_text", "-metadata:s:s:0", "language=eng", *trim, "-movflags", "+faststart", out_mp4]
        subprocess.run(cmd, check=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
