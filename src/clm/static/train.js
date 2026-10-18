/* CLM playground: the Train view.
 *
 * Four steps on one page: add labelled examples, describe the labels, train, see the
 * result and use it.  Talks to clm-serve's /v1/train/* routes; reuses app.js helpers
 * ($, esc, url, headers, post, answerHtml, toast, copy, shQuote, hydrate, paint, save, S).
 */
'use strict';

const T = {
  upload: null,          // /v1/train/inspect response
  textCol: '', labelCol: '', groupCol: '',
  question: '', kind: 'choice', kindTouched: false,
  options: {},           // label -> description
  name: '', nameTouched: false,
  testFrac: 0.2, epochs: 20,
  job: null, pollTimer: null,
  sample: false,
};

const SAMPLE = {
  file: 'sample/support-tickets.csv',
  filename: 'support-tickets.csv',
  question: 'What is this customer conversation primarily about?',
  options: {
    account: 'Login, plan changes, profile or access management.',
    billing: 'A charge, invoice, subscription or payment problem.',
    delivery: 'Shipping, fulfilment or delivery of a physical item.',
    refund: 'The customer is explicitly asking for money back.',
    technical: 'The product or service is not working as expected.',
  },
  tryText: 'customer: I was billed twice for my subscription this month and the second charge is still pending. Can you fix the invoice?',
};

const BOOL_WORDS = new Set(['true', 'false', 'yes', 'no', '1', '0', 'y', 'n', 't', 'f']);
const CATCH_ALL = /\b(other|others|misc|miscellaneous|none of|anything else|n\/a|general)\b/i;
const fmtInt = (n) => Number(n).toLocaleString();
const pct0 = (p) => Math.round(p * 100) + '%';

/* ───────────────────────────── server calls ───────────────────────────── */

async function api(method, path, body) {
  const r = await fetch(url(path), {
    method, headers: headers(!!body), body: body ? JSON.stringify(body) : undefined,
  });
  const text = await r.text();
  let json = null;
  try { json = JSON.parse(text); } catch (e) { /* plain-text error */ }
  if (!r.ok) {
    const d = (json && json.detail) || text || r.statusText;
    const err = new Error(typeof d === 'string' ? d : JSON.stringify(d));
    err.status = r.status;
    throw err;
  }
  return json;
}

function fileToB64(file) {
  return new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onload = () => resolve(String(fr.result).split(',', 2)[1] || '');
    fr.onerror = () => reject(new Error('the browser could not read that file'));
    fr.readAsDataURL(file);
  });
}

/* ───────────────────────────── derived state ───────────────────────────── */

const isTyped = () => T.upload && T.upload.format === 'typed';
function labelCounts() {
  if (!T.upload || !T.labelCol) return null;
  const st = T.upload.stats[T.labelCol];
  return st ? st.values : null;
}
function labels() {
  const c = labelCounts();
  return c ? Object.entries(c).sort((a, b) => b[1] - a[1]).map(([k]) => k) : [];
}
const boolLabels = () => { const l = labels(); return l.length > 0 && l.length <= 2 && l.every((k) => BOOL_WORDS.has(k.toLowerCase())); };

function autoKind() {
  if (!T.kindTouched) T.kind = boolLabels() ? 'noul' : 'choice';
  if (T.kind === 'noul' && !boolLabels()) T.kind = 'choice';
}

function step1Problems() {
  if (!T.upload) return ['Add a file of labelled examples.'];
  if (isTyped()) return [];
  const p = [];
  if (!T.textCol) p.push('Choose the column with the text.');
  if (!T.labelCol) p.push('Choose the column with the label.');
  if (T.textCol && T.textCol === T.labelCol) p.push('The text and the label must be different columns.');
  if (T.labelCol && T.textCol !== T.labelCol) {
    const c = labelCounts();
    if (!c) p.push(`“${T.labelCol}” has more than 100 different values, so it can't be a label column.`);
    else if (Object.keys(c).length < 2) p.push(`“${T.labelCol}” has only one label; you need at least two.`);
  }
  return p;
}
function step2Problems() {
  if (isTyped()) return [];
  const p = [];
  if (!T.question.trim()) p.push('Write the question the labels answer.');
  return p;
}
const readyToTrain = () => !step1Problems().length && !step2Problems().length;

function slug(s) {
  return String(s).toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').split('-').slice(0, 5).join('-') || 'model';
}

/* ───────────────────────────── step 1: data ───────────────────────────── */

async function useFile(file, prefill) {
  stopPolling();
  T.upload = null; T.job = null;
  $('#t-err-1').hidden = true;
  $('#t-busy-1').hidden = false;
  $('#t-pick').hidden = true;
  try {
    const b64 = await fileToB64(file);
    const info = await api('POST', '/v1/train/inspect', { filename: file.name, content_b64: b64 });
    Object.assign(T, {
      upload: info, textCol: info.suggested.text || '', labelCol: info.suggested.label || '', groupCol: '',
      question: prefill ? prefill.question : '', options: prefill ? { ...prefill.options } : {},
      kindTouched: false, nameTouched: false, sample: !!prefill, name: '',
    });
    autoKind();
  } catch (e) {
    $('#t-pick').hidden = false;
    const el = $('#t-err-1');
    el.innerHTML = `<b>Couldn't use that file.</b> ${esc(e.message)}`;
    el.hidden = false;
  } finally {
    $('#t-busy-1').hidden = true;
  }
  renderAll();
}

async function useSample() {
  try {
    const r = await fetch(url('/' + SAMPLE.file));
    if (!r.ok) throw new Error(`the sample file is missing (${r.status})`);
    const blob = await r.blob();
    await useFile(new File([blob], SAMPLE.filename, { type: 'text/csv' }), SAMPLE);
  } catch (e) {
    const el = $('#t-err-1');
    el.innerHTML = `<b>Couldn't load the sample.</b> ${esc(e.message)}`;
    el.hidden = false;
  }
}

function resetFile() {
  stopPolling();
  Object.assign(T, { upload: null, job: null, options: {}, question: '', sample: false });
  $('#t-pick').hidden = false;
  $('#t-file').value = '';
  renderAll();
}

function colSelect(id, value, cols, blank) {
  return `<select id="${id}">${blank ? `<option value="">${esc(blank)}</option>` : ''}${
    cols.map((c) => `<option value="${esc(c)}"${c === value ? ' selected' : ''}>${esc(c)}</option>`).join('')}</select>`;
}

function renderData() {
  const box = $('#t-data');
  if (!T.upload) { box.hidden = true; box.innerHTML = ''; return; }
  const u = T.upload;
  box.hidden = false;
  let html = `<div class="file-chip">
      <span class="name">${esc(u.filename)}</span>
      <span class="meta-txt">${fmtInt(u.rows)} rows · ${u.columns.length} columns${u.has_split ? ' · has a split column' : ''}</span>
      <button type="button" class="btn-ghost small" id="t-change" data-tip="Start over with another file. Nothing is trained until you press Train.">Use a different file</button>
    </div>`;

  if (isTyped()) {
    const qs = Object.entries(u.questions);
    html += `<ul class="tips"><li class="tip ok">This file already defines its questions and answers
      (${qs.length} question${qs.length === 1 ? '' : 's'}: ${qs.map(([k, q]) => `<code>${esc(k)}</code> ${esc(q.type || '')}`).join(', ')}).
      Training uses them as they are.</li></ul>`;
    box.innerHTML = html;
    $('#t-change').addEventListener('click', resetFile);
    return;
  }

  const cols = u.columns.filter((c) => c !== 'split');
  html += `<div class="col-pickers">
      <label><span class="lab-txt">Text to classify${tipQ('The column the model reads, such as the message or ticket body. Highlighted in teal in the preview.')}</span> ${colSelect('t-text-col', T.textCol, cols, 'Choose a column')}
        <small>What the model reads.</small></label>
      <label><span class="lab-txt">Correct label${tipQ('The column with the right answer for each row. Highlighted in red in the preview. Each different value becomes one label.')}</span> ${colSelect('t-label-col', T.labelCol, cols, 'Choose a column')}
        <small>The answer you want it to learn.</small></label>
      <label><span class="lab-txt">Group by <span class="opt">(optional)</span>${tipQ('If several rows come from the same customer, account or case, pick that column. Their rows then stay together, so the test never contains near-copies of training rows and the score is honest.')}</span> ${colSelect('t-group-col', T.groupCol, cols, 'None')}
        <small>For example a customer ID, so one customer's rows never end up in both training and testing.</small></label>
    </div>`;

  const heads = u.columns.map((c) => `<th class="${c === T.textCol ? 'is-text' : c === T.labelCol ? 'is-label' : ''}">${esc(c)}</th>`).join('');
  const body = u.preview.map((r) => `<tr>${u.columns.map((c) => `<td class="${c === T.textCol ? 'is-text' : c === T.labelCol ? 'is-label' : ''}">${esc(r[c] ?? '')}</td>`).join('')}</tr>`).join('');
  html += `<div class="t-table-wrap"><table class="t-table"><thead><tr>${heads}</tr></thead><tbody>${body}</tbody></table></div>`;

  const counts = labelCounts();
  const tips = [];
  if (counts) {
    const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
    html += `<div class="chips" aria-label="Labels and how many examples each has">${entries.map(([k, n]) =>
      `<span class="chip${n < 10 ? ' is-low' : ''}" data-tip="${fmtInt(n)} example${n === 1 ? '' : 's'} labelled “${esc(k)}”${n < 10 ? '. Too few to learn from reliably.' : '.'}" tabindex="0"><b>${esc(k)}</b><span>${fmtInt(n)}</span></span>`).join('')}</div>`;
    const low = entries.filter(([, n]) => n < 10).map(([k]) => k);
    if (low.length) tips.push(['warn', `${low.map((k) => `“${esc(k)}”`).join(', ')} ${low.length === 1 ? 'has' : 'have'} fewer than 10 examples. Aim for 30 or more per label, or the model will rarely pick ${low.length === 1 ? 'it' : 'them'}.`]);
    const top = entries[0][1] / u.rows;
    if (top > 0.7) tips.push(['warn', `${pct0(top)} of the rows are “${esc(entries[0][0])}”. Always answering that is already ${pct0(top)} accurate, so look for clear gains over it.`]);
  }
  for (const p of step1Problems()) tips.push(['bad', esc(p)]);
  const tst = u.stats[T.textCol];
  if (tst && tst.avg_len && tst.avg_len < 15) tips.push(['warn', 'The texts are very short. The model does best with a sentence or more of context.']);
  if (u.has_split) tips.push(['ok', 'Your file has a <code>split</code> column, so its <code>test</code> rows are used for testing.']);
  if (tips.length) html += `<ul class="tips">${tips.map(([c, t]) => `<li class="tip ${c}">${t}</li>`).join('')}</ul>`;

  box.innerHTML = html;
  $('#t-change').addEventListener('click', resetFile);
  $('#t-text-col').addEventListener('change', (e) => { T.textCol = e.target.value; renderAll(); });
  $('#t-label-col').addEventListener('change', (e) => { T.labelCol = e.target.value; T.kindTouched = false; autoKind(); renderAll(); });
  $('#t-group-col').addEventListener('change', (e) => { T.groupCol = e.target.value; renderAll(); });
}

/* ───────────────────────────── step 2: question ───────────────────────────── */

function optionHint(k) {
  const v = (T.options[k] || '').trim();
  if (T.kind === 'noul') return v ? ['', ''] : ['', `Optional: what a “${k}” example looks like.`];
  if (!v) return ['warn', `Not described yet, so the model only sees the word “${k}”.`];
  if (v.split(/\s+/).length < 4) return ['warn', 'Very short. Name the evidence: words or situations that show up in these examples.'];
  const dup = labels().find((o) => o !== k && (T.options[o] || '').trim().toLowerCase() === v.toLowerCase());
  if (dup) return ['warn', `Same as “${dup}”. Each label needs its own description.`];
  if (CATCH_ALL.test(v)) return ['warn', 'Catch-all wording matches everything a little. Say what these examples actually contain.'];
  return ['', ''];
}

function questionTips() {
  if (T.kind !== 'choice') return [];
  const missing = labels().filter((k) => !(T.options[k] || '').trim());
  const out = [];
  if (missing.length) out.push(['warn', `${missing.length} of ${labels().length} labels have no description. Training still works, but descriptions usually help a lot.`]);
  return out;
}

function renderQuestion() {
  const box = $('#t-question');
  if (!T.upload || step1Problems().length) { box.innerHTML = ''; return; }
  if (isTyped()) {
    box.innerHTML = '<ul class="tips"><li class="tip ok">Nothing to do here: your file defines the questions and their options.</li></ul>';
    return;
  }
  const canNoul = boolLabels();
  const qPlaceholder = T.kind === 'noul' ? 'e.g. This message needs a human agent to step in.'
                                         : 'e.g. What is this customer message about?';
  const keys = T.kind === 'noul' ? ['true', 'false'] : labels();
  const counts = labelCounts() || {};
  const countOf = (k) => {
    if (T.kind !== 'noul') return counts[k] || 0;
    return Object.entries(counts).filter(([lab]) => ['true', 'yes', '1', 'y', 't'].includes(lab.toLowerCase()) === (k === 'true'))
      .reduce((a, [, n]) => a + n, 0);
  };
  const rows = keys.map((k) => {
    const [cls, hint] = optionHint(k);
    const ph = T.kind === 'noul'
      ? (k === 'true' ? 'e.g. The customer is angry or asks for a manager.' : 'e.g. A routine question automation can answer.')
      : `What does “${k}” mean? Describe what these examples have in common.`;
    return `<div class="opt-row">
        <div class="opt-key"><b>${esc(k)}</b><span>${fmtInt(countOf(k))} examples</span></div>
        <div class="opt-input">
          <input type="text" id="t-opt-${esc(k)}" data-key="${esc(k)}" value="${esc(T.options[k] || '')}" placeholder="${esc(ph)}" aria-label="Description of ${esc(k)}" data-tip="The model compares each example with this description. Use the words and situations that appear in “${esc(k)}” examples.">
          <span class="opt-hint ${cls}" id="t-hint-${esc(k)}">${esc(hint)}</span>
        </div>
      </div>`;
  }).join('');
  box.innerHTML = `
    <label class="q-field"><span>What question do the labels answer?${tipQ('Asked about every example; the labels are its possible answers. It is added after each text, exactly as when you use the model later.')}</span>
      <input type="text" id="t-q" value="${esc(T.question)}" placeholder="${esc(qPlaceholder)}">
      <small>${T.kind === 'noul' ? 'Write it as a statement that is true or false for each example.' : 'Asked about every example; the answer is one of your labels.'}</small>
    </label>
    <div class="q-field"><span>Kind of answer</span>
      <div class="kind" role="group" aria-label="Kind of answer">
        <button type="button" data-kind="choice" class="${T.kind === 'choice' ? 'is-active' : ''}" aria-pressed="${T.kind === 'choice'}" data-tip="Each example has exactly one of your labels.">Pick one label</button>
        <button type="button" data-kind="noul" class="${T.kind === 'noul' ? 'is-active' : ''}" aria-pressed="${T.kind === 'noul'}" data-tip="${canNoul ? 'Each example is either true or false for your statement.' : 'Only available when the labels are true/false or yes/no.'}"${canNoul ? '' : ' disabled'}>Yes or no</button>
      </div>
    </div>
    <div class="opt-list">${rows}</div>
    <ul class="tips" id="t-q-tips"></ul>`;
  paintQuestionTips();
  $('#t-q').addEventListener('input', (e) => {
    T.question = e.target.value;
    if (!T.nameTouched) T.name = '';
    paintStepState(); renderTrainBox();
  });
  $$('.kind button', box).forEach((b) => b.addEventListener('click', () => {
    if (b.disabled) return;
    T.kind = b.dataset.kind; T.kindTouched = true; renderAll();
  }));
  $$('.opt-input input', box).forEach((inp) => inp.addEventListener('input', () => {
    T.options[inp.dataset.key] = inp.value;
    for (const k of keys) {            // a duplicate warning can appear on either row
      const [cls, hint] = optionHint(k);
      const h = document.getElementById('t-hint-' + k);
      if (h) { h.className = 'opt-hint ' + cls; h.textContent = hint; }
    }
    paintQuestionTips();
  }));
}

function paintQuestionTips() {
  const el = $('#t-q-tips');
  if (!el) return;
  const tips = questionTips();
  el.innerHTML = tips.map(([c, t]) => `<li class="tip ${c}">${esc(t)}</li>`).join('');
  el.hidden = !tips.length;
}

/* ───────────────────────────── step 3: train ───────────────────────────── */

function splitSentence() {
  const u = T.upload;
  if (u.has_split) return 'Trains on your file\'s <b>train</b> rows and tests on its <b>test</b> rows.';
  const test = Math.round(u.rows * T.testFrac);
  return `Holds back <b>${fmtInt(test)}</b> of ${fmtInt(u.rows)} examples (${pct0(T.testFrac)}) for testing${tipQ('Held-out examples: never shown to the model while training, so its score on them shows how it will do on new data.')}.
    The model never sees them while training, so the score reflects new data.${T.groupCol ? ` Rows with the same <b>${esc(T.groupCol)}</b> stay together.` : ''}`;
}

function renderTrainBox() {
  const box = $('#t-train');
  const running = T.job && ['queued', 'running'].includes(T.job.status);
  if (!T.upload || running) { box.innerHTML = ''; return; }
  const problems = [...step1Problems(), ...step2Problems()];
  const suggested = slug((T.upload && T.upload.filename.replace(/\.[^.]+$/, '')) || T.question || 'model');
  if (!T.nameTouched) T.name = suggested;
  box.innerHTML = `<div class="train-box">
      <label class="q-field"><span>Model name${tipQ('Letters, numbers and dashes. If the name is taken, a number is added.')}</span>
        <input type="text" id="t-name" value="${esc(T.name)}" spellcheck="false">
        <small>How you'll pick it in the playground and the API.</small>
      </label>
      <p class="split-line">${splitSentence()}</p>
      <details class="adv"><summary data-tip="Optional: how much to hold back for testing, and how long to train. The defaults suit most data.">Settings</summary>
        <div class="adv-grid">
          <label><span class="lab-txt">Held out for testing${tipQ('Share of examples set aside before training and used only to score the finished model. More gives a steadier score; less leaves more to learn from.')}</span>
            <select id="t-test-frac"${T.upload.has_split ? ' disabled title="Your file\'s split column decides this"' : ''}>
              ${[0.1, 0.2, 0.3].map((f) => `<option value="${f}"${f === T.testFrac ? ' selected' : ''}>${pct0(f)}</option>`).join('')}
            </select></label>
          <label><span class="lab-txt">Training rounds (at most)${tipQ('Each round goes over all training examples once. Training stops early when the check set stops improving, and the best round is kept.')}</span>
            <select id="t-epochs">${[10, 20, 40].map((n) => `<option value="${n}"${n === T.epochs ? ' selected' : ''}>${n}</option>`).join('')}</select></label>
        </div>
      </details>
      <div class="actions">
        <button type="button" class="btn-primary" id="t-go"${problems.length ? ' disabled' : ''} data-tip="${problems.length ? esc(problems[0]) : 'Start training on this server. You can keep using the playground meanwhile.'}">${T.job && T.job.status === 'done' ? 'Train again' : 'Train model'}</button>
        <span class="why-not">${problems.length ? esc(problems[0]) : 'Usually a few minutes; the encoder does most of the work.'}</span>
      </div>
    </div>`;
  $('#t-name').addEventListener('input', (e) => { T.name = e.target.value; T.nameTouched = true; });
  const tf = $('#t-test-frac');
  tf.addEventListener('change', () => { T.testFrac = Number(tf.value); $('.split-line').innerHTML = splitSentence(); });
  $('#t-epochs').addEventListener('change', (e) => { T.epochs = Number(e.target.value); });
  $('#t-go').addEventListener('click', startTraining);
}

async function startTraining() {
  $('#t-err-3').hidden = true;
  const go = $('#t-go');
  go.disabled = true;
  const body = {
    upload_id: T.upload.upload_id, filename: T.upload.filename, name: T.name,
    group_column: T.groupCol || null, test_frac: T.testFrac, epochs: T.epochs,
  };
  if (!isTyped()) {
    Object.assign(body, { text_column: T.textCol, label_column: T.labelCol, question: T.question.trim(), type: T.kind });
    const keys = T.kind === 'noul' ? ['true', 'false'] : labels();
    body.options = Object.fromEntries(keys.map((k) => [k, (T.options[k] || '').trim()]));
  }
  try {
    T.job = await api('POST', '/v1/train/jobs', body);
    renderAll();
    $('#t-step-3').scrollIntoView({ behavior: 'smooth', block: 'start' });
    startPolling();
  } catch (e) {
    const el = $('#t-err-3');
    el.innerHTML = `<b>Training didn't start.</b> ${esc(e.message)}`;
    el.hidden = false;
    go.disabled = false;
  }
}

function startPolling() {
  stopPolling();
  const tick = async () => {
    if (!T.job) return;
    try {
      T.job = await api('GET', '/v1/train/jobs/' + encodeURIComponent(T.job.id));
    } catch (e) { /* transient; keep polling */ }
    renderProgress();
    if (['done', 'failed', 'stopped'].includes(T.job.status)) {
      stopPolling();
      renderAll();
      loadJobs();
      refreshModels();
      if (T.job.status === 'done') {
        toast(`“${T.job.model}” is trained and ready`);
        $('#t-step-4').scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
      return;
    }
    T.pollTimer = setTimeout(tick, 1000);
  };
  T.pollTimer = setTimeout(tick, 400);
}
function stopPolling() { clearTimeout(T.pollTimer); T.pollTimer = null; }

function elapsed(job) {
  const end = job.finished || Date.now() / 1000;
  const s = Math.max(0, Math.round(end - (job.created || end)));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

function sparkline(history, base) {
  const pts = history.filter((h) => typeof h.val_acc === 'number');
  if (pts.length < 2) return '';
  const W = 360, H = 64, L = 4, R = 40, Tp = 6, B = 8;
  const maxE = Math.max(...pts.map((p) => p.epoch), 1);
  const x = (e) => L + (e / maxE) * (W - L - R);
  const y = (v) => Tp + (1 - v) * (H - Tp - B);
  const line = pts.map((p) => `${x(p.epoch).toFixed(1)},${y(p.val_acc).toFixed(1)}`).join(' ');
  const last = pts[pts.length - 1];
  const baseLine = typeof base === 'number'
    ? `<line class="base" x1="${L}" x2="${W - R}" y1="${y(base).toFixed(1)}" y2="${y(base).toFixed(1)}"/><text x="${W - R + 4}" y="${(y(base) + 3).toFixed(1)}">${pct0(base)}</text>` : '';
  return `<svg class="spark" viewBox="0 0 ${W} ${H}" role="img" aria-label="Validation accuracy by training round, now ${pct0(last.val_acc)}">
      <line class="axis" x1="${L}" x2="${W - R}" y1="${H - B}" y2="${H - B}"/>
      ${baseLine}
      <polyline class="line" points="${line}"/>
      <circle class="dot" cx="${x(last.epoch).toFixed(1)}" cy="${y(last.val_acc).toFixed(1)}" r="3"/>
      <text x="${(x(last.epoch) + 5).toFixed(1)}" y="${(y(last.val_acc) - 4).toFixed(1)}">${pct0(last.val_acc)}</text>
    </svg>`;
}

function renderProgress() {
  const box = $('#t-progress');
  const j = T.job;
  if (!j || !['queued', 'running', 'failed', 'stopped'].includes(j.status)) { box.hidden = true; box.innerHTML = ''; return; }
  box.hidden = false;
  if (j.status === 'failed' || j.status === 'stopped') {
    box.innerHTML = `<div class="t-err"><b>${j.status === 'failed' ? 'Training failed.' : 'Training stopped.'}</b>
      ${esc(j.error || 'You stopped it before it finished.')} ${j.status === 'failed' ? 'Fix the problem above and train again.' : ''}</div>`;
    return;
  }
  const p = j.progress || {};
  const order = ['queued', 'embedding', 'training', 'testing'];
  const stageOf = p.stage === 'baselines' ? 'training' : p.stage;
  const at = Math.max(0, order.indexOf(stageOf));
  const names = ['Reading examples', 'Reading the texts with Nemotron', 'Training', 'Testing'];
  const stages = names.map((n, i) => `<li class="${i < at ? 'is-past' : i === at ? 'is-now' : ''}">${n}</li>`).join('');
  let frac = 0, left = 'Getting ready…';
  if (stageOf === 'embedding' && p.total) {
    frac = p.done / p.total;
    left = `Read ${fmtInt(p.done)} of ${fmtInt(p.total)} texts`;
  } else if (stageOf === 'training' && p.epochs) {
    frac = (p.epoch || 0) / p.epochs;
    const best = Math.max(...j.history.map((h) => h.val_acc), 0);
    left = `Round ${p.epoch || 0} of up to ${p.epochs} · best so far ${pct0(best)} on the check set`;
  } else if (stageOf === 'testing') {
    frac = 1; left = 'Scoring on the held-out examples';
  }
  const base = j.baselines ? j.baselines.raw_val : undefined;
  box.innerHTML = `<div class="progress">
      <ol class="stages">${stages}</ol>
      <div class="pbar" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${Math.round(frac * 100)}"><div style="width:${(frac * 100).toFixed(1)}%"></div></div>
      <div class="pline"><span>${esc(left)}</span><span>${elapsed(j)}</span></div>
      ${sparkline(j.history || [], base)}
      ${base !== undefined && (j.history || []).length > 1 ? `<p class="note">Dashed line: the base model without training, on the same check set.${tipQ('The check set is 10% of the training examples, used to pick the best round. It is separate from the test examples, which are used only once, at the end.')}</p>` : ''}
      <div class="actions"><button type="button" class="btn-ghost" id="t-stop" data-tip="Stop now. Nothing is saved, and you can train again.">Stop training</button></div>
    </div>`;
  $('#t-stop').addEventListener('click', async () => {
    try { T.job = await api('POST', `/v1/train/jobs/${encodeURIComponent(j.id)}/stop`); toast('Stopping…'); }
    catch (e) { toast(e.message); }
  });
}

/* ───────────────────────────── step 4: results ───────────────────────────── */

function toPlayQuestion(id, q) {
  const crit = q.criteria || {};
  return {
    id, type: q.type, instructions: q.instructions || '',
    noulCrit: { true: (q.type === 'noul' && crit.true) || '', false: (q.type === 'noul' && crit.false) || '' },
    choiceCrit: q.type === 'choice' ? Object.entries(crit).map(([key, desc]) => ({ key, desc })) : undefined,
  };
}

function snippets(job, state) {
  const base = baseUrl() || location.origin;
  const q = { label: job.question };
  const py = `from clm import CLMClient

client = CLMClient(${JSON.stringify(base)})
r = client.system_one(
    state=${JSON.stringify(state || 'the text to classify')},
    questions=${JSON.stringify(q, null, 4).replace(/\n/g, '\n    ')},
    model=${JSON.stringify(job.model)},
)
print(r.answers["label"].${job.question.type === 'noul' ? 'noul' : 'choice'})`;
  const body = JSON.stringify({ state: state || 'the text to classify', model: job.model, questions: q });
  const auth = apiKey() ? ` \\\n  -H ${shQuote('Authorization: Bearer ' + apiKey())}` : '';
  const curl = `curl -s ${base}/v1/systemone \\\n  -H 'Content-Type: application/json'${auth} \\\n  -d ${shQuote(body)}`;
  return { py, curl };
}

function noiseBand(p, n) {
  return n ? Math.max(1, Math.round(100 * 1.96 * Math.sqrt(Math.max(p * (1 - p), 0.0025) / n))) : null;
}

function renderResults() {
  const box = $('#t-results');
  const j = T.job;
  if (!j || j.status !== 'done' || !j.result) { box.innerHTML = ''; return; }
  const r = j.result, n = (j.rows && j.rows.test) || 0;
  const gain = r.test_acc - r.majority_test;
  const weak = gain < 0.05;
  const band = noiseBand(r.test_acc, n);
  const perQ = Object.entries(r.per_question || {});
  const typed = !j.question;
  let html = `<div class="verdict${weak ? ' is-weak' : ''}">
      <div class="big">${weak ? 'Only ' : 'Gets '}<em>${pct0(r.test_acc)}</em> right</div>
      <p>on ${fmtInt(n)} held-out example${n === 1 ? '' : 's'} it never saw while training${band ? ` (give or take about ${band} points by chance${tipQ(`With ${n} test examples, the score can move by about ${band} points by chance alone. Compare models only on bigger differences.`)})` : ''}. Model <code>${esc(j.model)}</code>, trained in ${r.minutes < 1 ? Math.max(1, Math.round(r.minutes * 60)) + ' seconds' : r.minutes.toFixed(1) + ' minutes'}.</p>
    </div>
    <div class="cmp" aria-label="Accuracy compared with two baselines">
      ${[['Your trained model', r.test_acc, true, 'Your model, scored on the held-out test examples.'],
         ['Base model, no training', r.raw_test, false, 'The Nemotron encoder alone, comparing each text with your label descriptions (clm-raw). Shows what training added.'],
         ['Always the most common label', r.majority_test, false, 'What you would get by always answering with the most frequent label. A useful model must clearly beat this.']].map(([lab, v, me, tip]) =>
        `<div class="cmp-row${me ? ' is-model' : ''}" data-tip="${tip}" tabindex="0"><span class="lab">${lab}</span><div class="cmp-track"><div class="cmp-fill" style="width:${(v * 100).toFixed(1)}%"></div></div><span class="val">${pct0(v)}</span></div>`).join('')}
    </div>`;
  if (weak) {
    html += `<div class="callout"><b>This isn't much better than always guessing the most common label.</b> Before using it:
      <ul><li>Rewrite the label descriptions with the words that actually appear in each label's examples.</li>
      <li>Check that the text column holds what a person would need to decide the label.</li>
      <li>Add examples for the labels with few of them.</li></ul></div>`;
  }
  if (typed && perQ.length > 1) {
    html += `<div class="t-table-wrap"><table class="t-table"><thead><tr><th>Question</th><th class="n">Base model</th><th class="n">Trained</th></tr></thead><tbody>${
      perQ.sort((a, b) => b[1] - a[1]).map(([q, v]) => `<tr><td><code>${esc(q)}</code></td><td class="n">${pct0((r.raw_per_question || {})[q] || 0)}</td><td class="n">${pct0(v)}</td></tr>`).join('')}</tbody></table></div>`;
  }
  if (!typed) {
    html += `<div class="try"><h3>Try it</h3>
        <textarea id="t-try" rows="3" spellcheck="false" placeholder="Paste a new example to classify…" data-tip="Paste any new text, then press Classify. Nothing you type here is stored.">${esc(T.sample ? SAMPLE.tryText : '')}</textarea>
        <div class="actions"><button type="button" class="btn-primary" id="t-try-go" data-tip="Ask your model about the text above.">Classify</button>
          <span class="why-not">Trained models are very sure of themselves; treat the percentages as a ranking until you've calibrated.</span></div>
        <div id="t-try-out" class="try-out"></div>
      </div>`;
  }
  html += `<div class="use"><h3>Use it</h3>
      <p>It's live on this server as <code>${esc(j.model)}</code>. Pick it from the Model menu in the playground, or call the API.
        It is saved on the server and comes back after a restart.</p>
      <div class="actions">
        <button type="button" class="btn-primary" id="t-open" data-tip="Open the playground with this model, its question and your text filled in.">Open in playground</button>
        ${typed ? '' : '<button type="button" class="btn-ghost" id="t-copy-py" data-tip="Copy a Python snippet that calls this model.">Copy Python</button><button type="button" class="btn-ghost" id="t-copy-curl" data-tip="Copy a curl command that calls this model.">Copy curl</button>'}
      </div>
      ${j.checkpoint ? `<p class="note">Saved at <code>${esc(j.checkpoint_display || j.checkpoint)}</code>. Another server can load it with
        <code>clm-serve --encoder ${esc(j.encoder)} --model ${esc(j.model)}=${esc(j.checkpoint_display || j.checkpoint)}</code>.</p>` : ''}
    </div>`;
  box.innerHTML = html;

  const tryText = () => ($('#t-try') ? $('#t-try').value.trim() : '');
  if (!typed) {
    $('#t-try-go').addEventListener('click', async () => {
      const text = tryText();
      const out = $('#t-try-out');
      if (!text) { out.innerHTML = '<p class="why-not">Type or paste an example first.</p>'; return; }
      $('#t-try-go').disabled = true;
      try {
        const { json } = await post('/v1/systemone', { state: text, model: j.model, questions: { label: j.question } });
        out.innerHTML = answerHtml('label', j.question, json.answers.label);
      } catch (e) {
        out.innerHTML = `<div class="t-err"><b>Couldn't classify that.</b> ${esc(e.message)}</div>`;
      } finally {
        $('#t-try-go').disabled = false;
      }
    });
    $('#t-copy-py').addEventListener('click', () => copy(snippets(j, tryText()).py, 'Python snippet'));
    $('#t-copy-curl').addEventListener('click', () => copy(snippets(j, tryText()).curl, 'curl command'));
  }
  $('#t-open').addEventListener('click', async () => {
    await refreshModels();
    const state = typed ? '' : tryText();
    S = hydrate({ mode: 'ask', state, model: j.model,
                  questions: typed ? [] : [toPlayQuestion('label', j.question)] });
    paint(); save();
    showView('play');
    toast(`Loaded ${j.model} in the playground`);
  });
}

/* ───────────────────────────── models list ───────────────────────────── */

async function refreshModels() {
  try {
    const r = await fetch(url('/v1/models'), { headers: headers(false) });
    if (r.ok) fillModels((await r.json()).models || []);
  } catch (e) { /* the playground keeps its list */ }
}

async function loadJobs() {
  let jobs = [];
  try { jobs = (await api('GET', '/v1/train/jobs')).jobs || []; } catch (e) { return; }
  const box = $('#t-models');
  box.hidden = !jobs.length;
  const pill = (j) => j.status === 'done'
    ? `<span class="pill ${j.served ? 'ok' : ''}" data-tip="${j.served ? 'Loaded and answering requests on this server.' : esc('Not loaded: ' + (j.serve_error || 'trained with a different encoder than this server runs.'))}">${j.served ? 'Live' : 'Not loaded'}</span>`
    : j.status === 'running' || j.status === 'queued' ? '<span class="pill run" data-tip="Training right now.">Training</span>'
    : `<span class="pill bad" data-tip="${esc(j.error || 'Stopped before it finished.')}">${j.status === 'stopped' ? 'Stopped' : 'Failed'}</span>`;
  $('#t-models-body').innerHTML = `<div class="t-table-wrap"><table class="t-table">
      <thead><tr><th>Model</th><th>Status</th><th class="n">Accuracy</th><th class="n">Examples</th><th>Trained</th></tr></thead>
      <tbody>${jobs.map((j) => `<tr class="m-row" data-id="${esc(j.id)}" tabindex="0">
        <td><code>${esc(j.model)}</code></td><td>${pill(j)}</td>
        <td class="n">${j.result ? pct0(j.result.test_acc) : '–'}</td>
        <td class="n">${j.rows ? fmtInt(j.rows.train + j.rows.test) : '–'}</td>
        <td>${new Date((j.created || 0) * 1000).toLocaleString()}</td></tr>`).join('')}</tbody></table></div>
      <p class="note">Select a model to see its results.${jobs.some((j) => j.status === 'done' && !j.served) ? ' “Not loaded” models were trained with a different encoder than this server runs.' : ''}</p>`;
  $$('.m-row').forEach((row) => {
    const open = () => {
      const j = jobs.find((x) => x.id === row.dataset.id);
      if (!j) return;
      T.job = j;
      renderAll();
      if (['queued', 'running'].includes(j.status)) startPolling();
      $(j.status === 'done' ? '#t-step-4' : '#t-step-3').scrollIntoView({ behavior: 'smooth', block: 'start' });
    };
    row.addEventListener('click', open);
    row.addEventListener('keydown', (e) => { if (e.key === 'Enter') open(); });
  });
  const running = jobs.find((j) => ['queued', 'running'].includes(j.status));
  if (running && !T.pollTimer && (!T.job || T.job.id === running.id)) {
    T.job = running; renderAll(); startPolling();
  }
}

/* ───────────────────────────── page ───────────────────────────── */

function paintStepState() {
  const ok1 = !!T.upload && !step1Problems().length;
  const ok2 = ok1 && !step2Problems().length;
  const done = T.job && T.job.status === 'done';
  const running = T.job && ['queued', 'running', 'failed', 'stopped'].includes(T.job.status);
  $('#t-step-1').classList.toggle('is-done', ok1);
  $('#t-done-1').hidden = !ok1;
  $('#t-step-2').classList.toggle('is-locked', !ok1);
  $('#t-step-2').classList.toggle('is-done', ok2);
  $('#t-done-2').hidden = !ok2;
  $('#t-step-3').classList.toggle('is-locked', !(ok2 || running || done));
  $('#t-step-3').classList.toggle('is-done', !!done);
  $('#t-step-4').classList.toggle('is-locked', !done);
}

function renderAll() {
  renderData();
  renderQuestion();
  renderTrainBox();
  renderProgress();
  renderResults();
  paintStepState();
}

function showView(v) {
  const train = v === 'train' && !$('#viewswitch').hidden;
  $('#play-view').hidden = train;
  $('#train-view').hidden = !train;
  $$('.vs-btn').forEach((b) => {
    const on = b.dataset.view === (train ? 'train' : 'play');
    b.classList.toggle('is-active', on);
    b.setAttribute('aria-pressed', String(on));
  });
  try { localStorage.setItem('clm.playground.view', train ? 'train' : 'play'); } catch (e) { /* private mode */ }
  if (train) loadJobs();
}

// app.js calls this with every /health answer
window.onHealth = (health) => {
  const can = !!(health && health.training);
  const was = !$('#viewswitch').hidden;
  $('#viewswitch').hidden = !can;
  if (!can && !$('#train-view').hidden) showView('play');
  if (can && !was) {
    let want = location.hash === '#train' ? 'train' : '';
    if (!want) { try { want = localStorage.getItem('clm.playground.view') || ''; } catch (e) { /* ignore */ } }
    if (want === 'train') showView('train');
  }
};

function initTrain() {
  $$('.vs-btn').forEach((b) => b.addEventListener('click', () => showView(b.dataset.view)));
  const drop = $('#t-drop');
  $('#t-file').addEventListener('change', (e) => { if (e.target.files[0]) useFile(e.target.files[0]); });
  ['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('is-over'); }));
  ['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('is-over'); }));
  drop.addEventListener('drop', (e) => { const f = e.dataTransfer.files[0]; if (f) useFile(f); });
  $('#t-sample').addEventListener('click', useSample);
  renderAll();
}

document.addEventListener('DOMContentLoaded', initTrain);
