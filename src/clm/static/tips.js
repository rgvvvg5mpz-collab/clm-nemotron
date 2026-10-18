/* CLM playground: tooltips.
 *
 * Any element with data-tip="…" (or a title attribute, which is upgraded on first use so the
 * browser's own tooltip never doubles it) shows a bubble on hover and on keyboard focus.
 * Small "?" buttons (class tip-q) also open it on tap, for touch screens.  Esc closes it.
 * The bubble is announced through aria-describedby while it is open.
 */
'use strict';

(() => {
  let bubble = null, current = null, timer = null, pinned = false;

  function ensure() {
    if (bubble) return bubble;
    bubble = document.createElement('div');
    bubble.id = 'tip-bubble';
    bubble.className = 'tip-bubble';
    bubble.setAttribute('role', 'tooltip');
    bubble.hidden = true;
    document.body.appendChild(bubble);
    return bubble;
  }

  function targetOf(node) {
    const el = node && node.closest ? node.closest('[data-tip], [title]') : null;
    if (!el || el.tagName === 'OPTION' || el.closest('svg') === el) return null;
    if (el.hasAttribute('title')) {               // upgrade: one tooltip, not two
      const t = el.getAttribute('title');
      el.removeAttribute('title');
      if (t && !el.dataset.tip) el.dataset.tip = t;
    }
    return el.dataset.tip ? el : null;
  }

  function place(el) {
    const b = ensure();
    const r = el.getBoundingClientRect();
    const vw = document.documentElement.clientWidth, vh = document.documentElement.clientHeight;
    b.style.left = '0px'; b.style.top = '0px';
    const bw = b.offsetWidth, bh = b.offsetHeight, gap = 8, edge = 8;
    let top = r.bottom + gap;
    let below = true;
    if (top + bh > vh - edge && r.top - gap - bh > edge) { top = r.top - gap - bh; below = false; }
    let left = r.left + r.width / 2 - bw / 2;
    left = Math.max(edge, Math.min(left, vw - bw - edge));
    b.style.left = Math.round(left) + 'px';
    b.style.top = Math.round(top) + 'px';
    b.dataset.side = below ? 'below' : 'above';
    b.style.setProperty('--arrow-x', Math.round(Math.max(12, Math.min(bw - 12, r.left + r.width / 2 - left))) + 'px');
  }

  function show(el, now) {
    clearTimeout(timer);
    const go = () => {
      const b = ensure();
      if (current && current !== el) current.removeAttribute('aria-describedby');
      current = el;
      b.textContent = el.dataset.tip;
      b.hidden = false;
      el.setAttribute('aria-describedby', 'tip-bubble');
      place(el);
    };
    if (now) go(); else timer = setTimeout(go, 350);
  }

  function hide() {
    clearTimeout(timer);
    pinned = false;
    if (bubble) bubble.hidden = true;
    if (current) current.removeAttribute('aria-describedby');
    current = null;
  }

  document.addEventListener('mouseover', (e) => {
    if (pinned) return;
    const el = targetOf(e.target);
    if (el && el !== current) show(el, false);
    else if (!el && current && !current.contains(e.target)) hide();
  });
  document.addEventListener('mouseout', (e) => {
    if (pinned || !current) return;
    if (!e.relatedTarget || !current.contains(e.relatedTarget)) hide();
  });
  document.addEventListener('focusin', (e) => {
    const el = targetOf(e.target);
    if (el) show(el, true); else if (!pinned) hide();
  });
  document.addEventListener('focusout', () => { if (!pinned) hide(); });
  document.addEventListener('click', (e) => {
    const q = e.target.closest && e.target.closest('.tip-q');
    if (q) {
      e.preventDefault();
      const el = targetOf(q);
      if (pinned && current === el) { hide(); return; }
      if (el) { show(el, true); pinned = true; }
      return;
    }
    if (pinned) hide();
  }, true);
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') hide(); });
  window.addEventListener('scroll', () => { if (current && !bubble.hidden) place(current); }, true);
  window.addEventListener('resize', hide);
})();

/* A "?" button that explains a term; safe to drop into any innerHTML. */
function tipQ(text, label) {
  return `<button type="button" class="tip-q" data-tip="${esc(text)}" aria-label="${esc(label || 'What is this?')}">?</button>`;
}
