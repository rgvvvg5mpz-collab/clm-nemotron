"""Small video studio: narrate with macOS speech, drive the real app with Playwright, assemble with ffmpeg.

Used by the Crew Training video, the Marketing video and the user-guide screenshots.

How a recording works
  * The page is laid out at a logical size (say 1600x900) but rendered at twice that
    (`html { zoom: 2 }` in a 3200x1800 viewport), so text stays sharp after the final
    scale to 1080p and the post-production zooms have pixels to spare.
  * Frames come straight from Chrome's screencast (JPEG, every change, up to 30 fps),
    each stamped with the time it arrived, instead of Playwright's low-bitrate recorder.
  * Every narration clip is synthesised before recording starts, so its length is known.
    A scene starts its clip, runs its actions, then holds until the clip has finished.
  * Spotlights dim everything except one element and attach a callout that explains it.
    `focus()` also books a zoom on that region, applied later with ffmpeg's zoompan.
  * Assembly: frames -> constant 30 fps -> zooms -> scale -> H.264, with each narration
    clip placed at its scene's start, soft English subtitles, and an .srt beside the file.
"""
from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

VOICE = os.environ.get("STUDIO_VOICE", "Ava (Premium)")
RATE = os.environ.get("STUDIO_RATE", "176")          # words per minute for `say`
FPS = 30
DRY = bool(os.environ.get("STUDIO_DRY"))        # STUDIO_DRY=1: no speech, no frames, no waits; checks every step


# ----------------------------------------------------------------------------- audio
def tts(text: str, path: str) -> float:
    """Speak `text` into `path` (.aiff); returns its duration in seconds."""
    clean = re.sub(r"\s+", " ", text).strip()
    if DRY:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono",
                        "-t", "0.1", path], check=True)
        return 0.1
    subprocess.run(["say", "-v", VOICE, "-r", RATE, "-o", path, clean], check=True)
    return duration(path)


def duration(path: str) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", path],
                         capture_output=True, text=True, check=True).stdout
    return float(json.loads(out)["format"]["duration"])


# ----------------------------------------------------------------------------- overlay
# Everything the studio draws lives inside the page, in logical pixels. The page may be
# zoomed (html { zoom }), in which case client rects come back multiplied by the zoom and
# are divided out before positioning.
OVERLAY_JS = r"""
([zoom, hideCursor]) => {
  if (zoom && document.documentElement.style.zoom !== String(zoom)) document.documentElement.style.zoom = String(zoom);
  if (window.__studio) { window.__studio.cursorVisible(!hideCursor); return; }
  const FONT = '-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif';
  const css = `
  #studio-cap{position:fixed;left:50%;bottom:28px;transform:translateX(-50%);z-index:2147483646;max-width:min(1180px,88vw);
    background:rgba(14,15,20,.88);color:#fff;font:500 22px/1.4 ${FONT};padding:13px 24px;border-radius:12px;
    box-shadow:0 10px 34px rgba(0,0,0,.4);text-align:center;transition:opacity .3s;backdrop-filter:blur(6px)}
  #studio-step{position:fixed;right:22px;top:64px;z-index:2147483646;background:#b1040e;color:#fff;border-radius:999px;
    font:700 14px ${FONT};letter-spacing:.02em;padding:7px 14px;box-shadow:0 4px 14px rgba(0,0,0,.25);transition:opacity .3s}
  #studio-cursor{position:fixed;z-index:2147483647;width:24px;height:24px;margin:-3px 0 0 -3px;pointer-events:none;
    transition:left .6s cubic-bezier(.25,.8,.3,1),top .6s cubic-bezier(.25,.8,.3,1),opacity .3s;filter:drop-shadow(0 2px 3px rgba(0,0,0,.4))}
  #studio-cursor.click::after{content:"";position:absolute;left:-13px;top:-13px;width:48px;height:48px;border-radius:50%;
    border:3px solid rgba(177,4,14,.8);animation:studio-ping .55s ease-out forwards}
  @keyframes studio-ping{from{transform:scale(.3);opacity:1}to{transform:scale(1.25);opacity:0}}
  #studio-spot{position:fixed;z-index:2147483644;pointer-events:none;border-radius:10px;border:2.5px solid #e9a200;
    box-shadow:0 0 0 9999px rgba(8,10,16,.56),0 0 0 5px rgba(233,162,0,.28),0 0 28px rgba(233,162,0,.35);
    transition:left .55s cubic-bezier(.3,.7,.3,1),top .55s cubic-bezier(.3,.7,.3,1),width .55s cubic-bezier(.3,.7,.3,1),
      height .55s cubic-bezier(.3,.7,.3,1),opacity .4s}
  #studio-callout{position:fixed;z-index:2147483645;pointer-events:none;width:360px;background:#15161c;color:#f4f2ef;
    border-radius:12px;padding:14px 18px 15px;box-shadow:0 14px 40px rgba(0,0,0,.45);border:1px solid rgba(255,255,255,.08);
    font:400 15.5px/1.45 ${FONT};transition:opacity .35s,left .55s cubic-bezier(.3,.7,.3,1),top .55s cubic-bezier(.3,.7,.3,1)}
  #studio-callout b{display:block;font:700 17px/1.25 ${FONT};color:#ffd166;margin-bottom:6px}
  #studio-callout .arr{position:absolute;width:16px;height:16px;background:#15161c;transform:rotate(45deg);
    border:1px solid rgba(255,255,255,.08);border-right:0;border-bottom:0}
  #studio-card{position:fixed;inset:0;z-index:2147483640;display:flex;align-items:center;justify-content:center;
    background:#0f1115;color:#f3f1ef;font-family:${FONT};transition:opacity .4s}
  #studio-card .inner{width:min(1240px,88vw)}
  #tip-bubble{display:none!important}   /* the app's own hover tooltips: the callouts explain things here */
  `;
  const st = document.createElement('style'); st.textContent = css; document.head.appendChild(st);
  const mk = (id) => { const d = document.createElement('div'); d.id = id; d.style.opacity = 0; document.body.appendChild(d); return d; };
  const cap = mk('studio-cap'), step = mk('studio-step'), spot = mk('studio-spot'), callout = mk('studio-callout');
  const cur = mk('studio-cursor'); cur.style.opacity = hideCursor ? 0 : 1;
  cur.innerHTML = '<svg viewBox="0 0 24 24" width="24" height="24"><path d="M3 2l7 19 2.6-7.4L20 11z" fill="#111" stroke="#fff" stroke-width="1.6" stroke-linejoin="round"/></svg>';
  cur.style.left = '50vw'; cur.style.top = '60vh';
  const Z = () => parseFloat(getComputedStyle(document.documentElement).zoom) || 1;
  const vw = () => window.innerWidth / Z(), vh = () => window.innerHeight / Z();
  const rectOf = (sel, pad) => {
    const el = typeof sel === 'string' ? document.querySelector(sel) : sel; if (!el) return null;
    const r = el.getBoundingClientRect(), z = Z();
    return {left: r.left / z - pad, top: r.top / z - pad, width: r.width / z + 2 * pad, height: r.height / z + 2 * pad};
  };
  const px = (el, r) => Object.assign(el.style, {left: r.left + 'px', top: r.top + 'px', width: r.width + 'px', height: r.height + 'px'});
  let spotRect = null, chromeHidden = false, capText = '', stepText = '';
  let tracking = null;       // {sel, pad}: the spotlight follows its element if the page moves or re-renders
  const same = (a, b) => a && b && ['left', 'top', 'width', 'height'].every(k => Math.abs(a[k] - b[k]) < 1);
  (function track() {
    if (tracking) {
      const r = rectOf(tracking.sel, tracking.pad);
      if (r && !same(r, spotRect)) { px(spot, r); spotRect = r; if (callout.style.opacity === '1') place(r); }
    }
    requestAnimationFrame(track);
  })();
  const showChrome = () => { cap.style.opacity = capText && !chromeHidden ? 1 : 0; step.style.opacity = stepText && !chromeHidden ? 1 : 0; };
  function place(r) {
    // the callout goes beside the spotlight: right, left, below or above, whichever has room
    const W = 360, gap = 18, H = callout.offsetHeight || 110, w = vw(), h = vh();
    let left, top, side;
    if (r.left + r.width + gap + W < w - 12) { side = 'left'; left = r.left + r.width + gap; top = r.top + r.height / 2 - H / 2; }
    else if (r.left - gap - W > 12) { side = 'right'; left = r.left - gap - W; top = r.top + r.height / 2 - H / 2; }
    else if (r.top + r.height + gap + H < h - 90) { side = 'top'; left = r.left + r.width / 2 - W / 2; top = r.top + r.height + gap; }
    else { side = 'bottom'; left = r.left + r.width / 2 - W / 2; top = r.top - gap - H; }
    left = Math.max(12, Math.min(left, w - W - 12)); top = Math.max(12, Math.min(top, h - H - 90));
    callout.style.left = left + 'px'; callout.style.top = top + 'px';
    const a = callout.querySelector('.arr'); if (!a) return;
    const cx = r.left + r.width / 2 - left, cy = r.top + r.height / 2 - top;
    const clampX = Math.max(14, Math.min(cx, W - 30)), clampY = Math.max(14, Math.min(cy, H - 30));
    if (side === 'left') Object.assign(a.style, {left: '-9px', top: clampY - 8 + 'px', transform: 'rotate(-45deg)'});
    if (side === 'right') Object.assign(a.style, {left: W - 9 + 'px', top: clampY - 8 + 'px', transform: 'rotate(135deg)'});
    if (side === 'top') Object.assign(a.style, {left: clampX - 8 + 'px', top: '-9px', transform: 'rotate(45deg)'});
    if (side === 'bottom') Object.assign(a.style, {left: clampX - 8 + 'px', top: H - 9 + 'px', transform: 'rotate(225deg)'});
    return {left, top, width: W, height: H};
  }
  window.__studio = {
    caption(t) { capText = t || ''; cap.textContent = capText; showChrome(); },
    step(t) { stepText = t || ''; step.textContent = stepText; showChrome(); },
    chrome(v) { chromeHidden = !v; showChrome(); },     // hidden while zoomed in: a fixed bar would be cropped
    move(x, y) { cur.style.left = x / Z() + 'px'; cur.style.top = y / Z() + 'px'; },
    cursorVisible(v) { cur.style.opacity = v ? 1 : 0; },
    click() { cur.classList.remove('click'); void cur.offsetWidth; cur.classList.add('click'); },
    spot(sel, title, text, pad) {
      const r = rectOf(sel, pad == null ? 8 : pad);
      if (!r) { this.unspot(); return null; }
      const fresh = spot.style.opacity === '0' || spot.style.opacity === '';
      if (fresh) { spot.style.transition = 'opacity .4s'; px(spot, r); void spot.offsetWidth; spot.style.transition = ''; }
      px(spot, r); spot.style.opacity = 1; spotRect = r; tracking = {sel, pad: pad == null ? 8 : pad};
      let c = null;
      if (title || text) {
        callout.innerHTML = '<div class="arr"></div>' + (title ? '<b></b>' : '') + '<span></span>';
        if (title) callout.querySelector('b').textContent = title;
        callout.querySelector('span').textContent = text || '';
        if (fresh) { callout.style.transition = 'opacity .35s'; place(r); void callout.offsetWidth; callout.style.transition = ''; }
        c = place(r); callout.style.opacity = 1;
      } else callout.style.opacity = 0;
      return {spot: r, callout: c, view: {width: vw(), height: vh()}};
    },
    unspot() { spot.style.opacity = 0; callout.style.opacity = 0; spotRect = null; tracking = null; },
    rect(sel, pad) { const r = rectOf(sel, pad || 0); return r ? {...r, view: {width: vw(), height: vh()}} : null; },
    card(html) {
      let c = document.getElementById('studio-card');
      if (!html) { cur.style.opacity = hideCursor ? 0 : 1; if (c) { c.style.opacity = 0; setTimeout(() => c.remove(), 450); } return; }
      this.unspot(); this.step(''); this.caption(''); cur.style.opacity = 0;    // a card is a clean slide
      if (!c) { c = document.createElement('div'); c.id = 'studio-card'; c.style.opacity = 0; document.body.appendChild(c); }
      c.innerHTML = '<div class="inner">' + html + '</div>'; requestAnimationFrame(() => c.style.opacity = 1);
    },
  };
}
"""


# ----------------------------------------------------------------------------- capture
class Capture:
    """Chrome's screencast, saved frame by frame with arrival times (seconds since `t0`)."""

    def __init__(self, context, page, workdir: str, t0: float, width: int, height: int, quality: int = 90):
        self.dir = os.path.join(workdir, "frames")
        os.makedirs(self.dir, exist_ok=True)
        self.frames: list[tuple[float, str]] = []
        self.t0 = t0
        self.cdp = context.new_cdp_session(page)
        if DRY:
            return
        self.cdp.on("Page.screencastFrame", self._on_frame)
        self.cdp.send("Page.startScreencast", {"format": "jpeg", "quality": quality, "maxWidth": width,
                                               "maxHeight": height, "everyNthFrame": 2})

    def _on_frame(self, ev):
        t = time.monotonic() - self.t0
        path = os.path.join(self.dir, f"{len(self.frames):06d}.jpg")
        with open(path, "wb") as f:
            f.write(base64.b64decode(ev["data"]))
        self.frames.append((t, path))
        self.cdp.send("Page.screencastFrameAck", {"sessionId": ev["sessionId"]})

    def stop(self):
        try:
            self.cdp.send("Page.stopScreencast")
        except Exception:
            pass


# ----------------------------------------------------------------------------- recorder
class Recorder:
    """One continuous recording of a Playwright page, with narrated scenes, spotlights and zooms.

    `narrations` maps scene keys to narration text; every clip is synthesised up front.
    """

    def __init__(self, context, page, workdir: str, narrations: dict[str, str],
                 logical: tuple[int, int] = (1600, 900), zoom: int = 2, cursor: bool = True):
        self.page, self.workdir = page, workdir
        self.logical, self.zoom, self.cursor = logical, zoom, cursor
        self.scenes: list[dict] = []
        self.zooms: list[dict] = []
        self.audio_dir = os.path.join(workdir, "audio")
        os.makedirs(self.audio_dir, exist_ok=True)
        self.clips = self._synthesise(narrations)
        self.t0 = time.monotonic()
        self.capture = Capture(context, page, workdir, self.t0, logical[0] * zoom, logical[1] * zoom)

    def _synthesise(self, narrations: dict[str, str]) -> dict[str, tuple[str, float, str]]:
        def one(item):
            key, text = item
            path = os.path.join(self.audio_dir, f"{key}.aiff")
            return key, (path, tts(text, path), text)
        with ThreadPoolExecutor(max_workers=4) as pool:
            return dict(pool.map(one, narrations.items()))

    def now(self) -> float:
        return time.monotonic() - self.t0

    # --- overlay plumbing
    def overlay(self):
        self.page.evaluate(OVERLAY_JS, [self.zoom, not self.cursor])

    def js(self, expr: str, *args):
        self.overlay()
        return self.page.evaluate(expr, *args) if args else self.page.evaluate(expr)

    # --- scene timing
    def scene(self, key: str, caption: str = "", step: str | None = None) -> dict:
        """Start the scene `key`: play its narration from now and show the caption."""
        path, dur, text = self.clips[key]
        s = {"key": key, "start": self.now(), "audio": path, "dur": dur, "text": text, "caption": caption}
        self.scenes.append(s)
        print(f"[studio] {int(s['start'] // 60)}:{int(s['start'] % 60):02d}  scene {key}", flush=True)
        self.overlay()
        self.page.evaluate("t => window.__studio.caption(t)", caption)
        if step is not None:
            self.page.evaluate("t => window.__studio.step(t)", step)
        return s

    def hold(self, extra: float = 0.6):
        """Wait until the current scene's narration has finished (plus a short breath)."""
        s = self.scenes[-1]
        left = s["start"] + s["dur"] + extra - self.now()
        if left > 0 and not DRY:
            self.page.wait_for_timeout(int(left * 1000))
        s["end"] = self.now()
        self.unzoom()

    def wait(self, seconds: float):
        self.page.wait_for_timeout(int((0.05 if DRY else seconds) * 1000))

    def _target(self, selector: str, what: str):
        """The first element for `selector`, scrolled into view; None (with a note) when it is missing or hidden,
        so a changed page never aborts a recording."""
        el = self.page.locator(selector).first
        try:
            if el.count() == 0:
                print(f"[studio] {what}: nothing matches {selector!r}", flush=True)
                return None
            el.scroll_into_view_if_needed(timeout=2500)
        except Exception:
            try:
                self.page.evaluate("s => document.querySelector(s).scrollIntoView({block: 'center'})", selector)
            except Exception:
                pass
            if not el.is_visible():
                print(f"[studio] {what}: {selector!r} is not visible", flush=True)
                return None
        return el

    # --- actions with a visible cursor
    def point(self, selector: str, wait: float = 0.7):
        self.overlay()
        el = self._target(selector, "point")
        if el is None:
            return self.page.locator(selector).first
        box = el.bounding_box()
        if box:
            self.page.evaluate("([x, y]) => window.__studio.move(x, y)", [box["x"] + box["width"] / 2, box["y"] + box["height"] / 2])
        self.wait(wait)
        return el

    def click(self, selector: str, wait: float = 0.6):
        el = self.point(selector)
        self.page.evaluate("() => window.__studio.click()")
        el.click()
        self.wait(wait)

    def type(self, selector: str, text: str, delay: int = 28, clear: bool = True):
        el = self.point(selector)
        el.click()
        if clear:
            el.fill("")
        el.type(text, delay=delay)

    def scroll_to(self, selector: str, block: str = "start"):
        self.page.evaluate("([s, b]) => document.querySelector(s).scrollIntoView({behavior: 'smooth', block: b})", [selector, block])
        self.wait(0.75)

    def card(self, html: str | None):
        self.unzoom()
        self.overlay()
        self.page.evaluate("h => window.__studio.card(h)", html)

    # --- spotlight and zoom
    def spot(self, selector: str, title: str = "", text: str = "", pad: int = 8, wait: float = 0.8, zoom: float | None = None):
        """Dim everything but `selector`, explain it in a callout, and (optionally) zoom on it.

        `zoom` is the magnification; None picks one that keeps the element and its callout in view."""
        self.overlay()
        el = self._target(selector, "spot")
        if el is None:
            self.unspot()
            self.wait(wait)
            return self.page.locator(selector).first
        self.page.wait_for_timeout(120)
        box = el.bounding_box()
        if box:
            self.page.evaluate("([x, y]) => window.__studio.move(x, y)", [box["x"] + box["width"] / 2, box["y"] + box["height"] / 2])
        geo = self.page.evaluate("([s, a, b, p]) => window.__studio.spot(s, a, b, p)", [selector, title, text, pad])
        if geo and zoom != 1:
            self._zoom_on(geo, zoom)
        self.wait(wait)
        return el

    def focus(self, selector: str, title: str = "", text: str = "", **kw):
        """`spot()` that always zooms (unless kw zoom=1)."""
        return self.spot(selector, title, text, **kw)

    def unspot(self):
        self.overlay()
        self.page.evaluate("() => window.__studio.unspot()")
        self.unzoom()

    def _zoom_on(self, geo: dict, zoom: float | None):
        W, H = geo["view"]["width"], geo["view"]["height"]
        rects = [geo["spot"]] + ([geo["callout"]] if geo.get("callout") else [])
        x0 = min(r["left"] for r in rects) - 24; y0 = min(r["top"] for r in rects) - 24
        x1 = max(r["left"] + r["width"] for r in rects) + 24; y1 = max(r["top"] + r["height"] for r in rects) + 24
        z = zoom or min(1.7, W / max(x1 - x0, 1), H / max(y1 - y0, 1))
        z = max(1.0, min(z, 2.0))
        if z < 1.08:
            self.unzoom()
            return
        self.unzoom()
        self.zooms.append({"start": self.now(), "z": round(z, 3), "cx": round((x0 + x1) / 2 / W, 4), "cy": round((y0 + y1) / 2 / H, 4)})
        self.page.evaluate("() => window.__studio.chrome(false)")

    def zoom_to(self, selector: str, zoom: float = 1.6, pad: int = 20):
        """Zoom on a region without a spotlight."""
        self.overlay()
        geo = self.page.evaluate("([s, p]) => window.__studio.rect(s, p)", [selector, pad])
        if geo:
            self._zoom_on({"spot": geo, "view": geo["view"]}, zoom)

    def unzoom(self):
        if self.zooms and "end" not in self.zooms[-1]:
            self.zooms[-1]["end"] = self.now()
            try:
                self.page.evaluate("() => window.__studio && window.__studio.chrome(true)")
            except Exception:      # the page may already be closed
                pass

    def finish(self) -> float:
        end = self.now() + 0.4
        self.page.wait_for_timeout(400)
        self.unzoom()
        self.capture.stop()
        return end


# ----------------------------------------------------------------------------- assembly
def srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# spellings written for the speech engine, shown normally in subtitles
SPOKEN = {"C L M": "CLM", "C S V": "CSV", "I D": "ID", "localhost, port eighty-seven hundred": "localhost:8700",
          "J SON": "JSON", "A P I": "API"}


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


def _ease(a: float, b: float, ramp: float = 0.7) -> str:
    """0 -> 1 -> 0 over [a, b] with smooth ramps of `ramp` seconds, as an ffmpeg expression in in_time."""
    u = f"min(1,max(0,min((in_time-{a:.3f})/{ramp},({b:.3f}-in_time)/{ramp})))"
    return f"({u}*{u}*(3-2*{u}))"


def zoompan_filter(zooms: list[dict], width: int, height: int) -> str:
    """One zoompan filter applying every booked zoom, eased in and out, panning to its centre."""
    segs = [z for z in zooms if z.get("end", 0) - z["start"] > 1.0]
    if not segs:
        return ""
    z_expr, cx_expr, cy_expr = "1", "0.5", "0.5"
    for z in reversed(segs):
        e = _ease(z["start"], z["end"])
        cond = f"between(in_time,{z['start']:.3f},{z['end']:.3f})"
        z_expr = f"if({cond},1+{z['z'] - 1:.3f}*{e},{z_expr})"
        cx_expr = f"if({cond},0.5+{z['cx'] - 0.5:.4f}*{e},{cx_expr})"
        cy_expr = f"if({cond},0.5+{z['cy'] - 0.5:.4f}*{e},{cy_expr})"
    x = f"clip(({cx_expr})*iw-iw/zoom/2,0,iw-iw/zoom)"
    y = f"clip(({cy_expr})*ih-ih/zoom/2,0,ih-ih/zoom)"
    return f"zoompan=z='{z_expr}':x='{x}':y='{y}':d=1:fps={FPS}:s={width}x{height}"


def assemble(rec: Recorder, out_mp4: str, srt: str, end: float, size: tuple[int, int] = (1920, 1080),
             offset: float = 0.0) -> None:
    """Frames + narration + subtitles -> H.264 MP4 at `size`."""
    frames = rec.capture.frames
    if DRY:
        print(f"[studio] dry run: {len(rec.scenes)} scenes, {len(rec.zooms)} zooms, {end:.0f}s; nothing written")
        return
    if not frames:
        raise RuntimeError("no frames were captured")
    W, H = rec.logical[0] * rec.zoom, rec.logical[1] * rec.zoom
    lst = os.path.join(rec.workdir, "frames.txt")
    with open(lst, "w") as f:
        times = [0.0] + [t for t, _ in frames[1:]] + [end]       # the first frame stands in from t=0
        for i, (_, path) in enumerate(frames):
            d = max(times[i + 1] - times[i], 1 / FPS / 2)
            f.write(f"file '{path}'\nduration {d:.4f}\n")
        f.write(f"file '{frames[-1][1]}'\n")
    vf = [f"fps={FPS}", f"scale={W}:{H}"]
    zp = zoompan_filter(rec.zooms, W, H)
    if zp:
        vf.append(zp)
    vf += [f"scale={size[0]}:{size[1]}:flags=lanczos", "format=yuv420p"]

    inputs, filters = ["-f", "concat", "-safe", "0", "-i", lst], []
    for i, s in enumerate(rec.scenes):
        inputs += ["-i", s["audio"]]
        ms = max(0, int((s["start"] + offset) * 1000))
        filters.append(f"[{i + 1}:a]aresample=48000,adelay={ms}|{ms}[a{i}]")
    mix = "".join(f"[a{i}]" for i in range(len(rec.scenes)))
    filters.append(f"{mix}amix=inputs={len(rec.scenes)}:normalize=0,apad[aout]")
    filters.append("[0:v]" + ",".join(vf) + "[vout]")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", *inputs, "-i", srt,
           "-filter_complex", ";".join(filters),
           "-map", "[vout]", "-map", "[aout]", "-map", f"{len(rec.scenes) + 1}:s",
           "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", "-r", str(FPS),
           "-c:a", "aac", "-b:a", "192k", "-c:s", "mov_text", "-metadata:s:s:0", "language=eng",
           "-t", f"{end:.2f}", "-movflags", "+faststart", out_mp4]
    subprocess.run(cmd, check=True)


def mkwork(prefix: str) -> str:
    return tempfile.mkdtemp(prefix=prefix)


def cleanup(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)
