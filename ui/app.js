'use strict';
/*
 * UI сервиса проверки кассовой дисциплины (R6, R10). Без сборки и CDN.
 * Сценарий: видео → POST /api/v1/analyze (X-API-Key) → опрос GET /api/v1/jobs/{id}/details раз в 2 с → вердикт и карточки.
 * Демо без бэкенда: ?mock=1 (с нарушениями), ?mock=clean (без нарушений), ?mock=fail (ошибка); ответ — ui/mock_details.json.
 * API — только тот же origin, что и страница: сохранённый X-API-Key не уходит на чужой адрес.
 */

const $ = (sel, root = document) => root.querySelector(sel);

const params = new URLSearchParams(location.search);
const MOCK = params.get('mock');
const API = '/api/v1';
const KEY_STORE = 'cashcheck.apiKey';
const POLL_MS = 2000;
const NET_RETRIES = 3;
// сколько ждём ответа, прежде чем считать связь потерянной
const HEALTH_TIMEOUT_MS = 10_000;
const POLL_TIMEOUT_MS = 30_000; // details с кадрами весит до нескольких МБ
const VIDEO_TIMEOUT_MS = 180_000; // исходник ~29 МБ, на мобильном интернете — долго
const UPLOAD_STALL_MS = 30_000; // файл не двигается — сервер сам закрывает загрузку через 15 с
const UPLOAD_TIMEOUT_MS = 240_000; // загрузка + ответ целиком; у сервера лимит загрузки 180 с
const MAX_BYTES = 200 * 1024 * 1024;
const UPLOAD_SHARE = 0.15; // доля общей полосы, которую занимает загрузка файла
const STEPS = ['upload', 'decode', 'detect', 'verify', 'report'];
const STAGE_TITLE = {
  upload: 'Загружаем видео',
  queued: 'Видео в очереди',
  decode: 'Раскадровка',
  detect: 'Ищем события',
  verify: 'Проверяем по правилам',
  report: 'Готовим отчёт',
};

const ICON_BAD = '<svg viewBox="0 0 40 40" fill="none" stroke="currentColor" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"><path d="M20 4 37 34H3z"/><path d="M20 15v9"/><circle cx="20" cy="29" r="0.6" fill="currentColor"/></svg>';
const ICON_GOOD = '<svg viewBox="0 0 40 40" fill="none" stroke="currentColor" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"><circle cx="20" cy="20" r="16"/><path d="m12.5 20.5 5 5 10-11"/></svg>';

const el = {
  service: $('#service'), serviceText: $('#service-text'),
  upload: $('#view-upload'), progress: $('#view-progress'), failure: $('#view-failure'), result: $('#view-result'),
  key: $('#api-key'), keyToggle: $('#key-toggle'),
  drop: $('#drop'), file: $('#file'), fileRow: $('#file-row'), fileName: $('#file-name'), fileSize: $('#file-size'),
  start: $('#start'), demo: $('#demo'), formError: $('#form-error'),
  progressTitle: $('#progress-title'), elapsed: $('#elapsed'), bar: $('#bar'), barFill: $('#bar-fill'),
  steps: $('#steps'), progressNote: $('#progress-note'), cancel: $('#cancel'),
  failureText: $('#failure-text'), failureDetail: $('#failure-detail'), retry: $('#retry'), chooseOther: $('#choose-other'),
  verdict: $('#verdict'), verdictIcon: $('#verdict-icon'), verdictTitle: $('#verdict-title'),
  verdictText: $('#verdict-text'), verdictMeta: $('#verdict-meta'), badgeMock: $('#badge-mock'),
  playerPanel: $('#player-panel'), player: $('#player'), playerNote: $('#player-note'),
  timeline: $('#timeline'), timelineLabel: $('#timeline-label'), playhead: $('#playhead'), ticks: $('#ticks'),
  cases: $('#cases'), again: $('#again'),
  tplCase: $('#tpl-case'), tplRule: $('#tpl-rule'),
  lightbox: $('#lightbox'), lightboxImg: $('#lightbox-img'), lightboxCaption: $('#lightbox-caption'), lightboxClose: $('#lightbox-close'),
};

const state = {
  file: null,
  objectUrl: null,
  jobId: params.get('job'),
  xhr: null,
  pollTimer: 0,
  clockTimer: 0,
  startedAt: 0,
  netFails: 0,
  lastError: null,
  duration: 0,
  runId: 0, // растёт при отмене, чтобы устаревшие ответы не трогали экран
};

/* ---------- мелочи ---------- */

class UiError extends Error {
  /** retry: 'upload' — загрузить файл заново, 'poll' — продолжить опрос той же проверки, 'none' — только другой файл */
  constructor(message, { detail = '', retry = 'upload', field = '' } = {}) {
    super(message);
    this.detail = detail;
    this.retry = retry;
    this.field = field; // 'key' — исправить можно только на экране загрузки, в поле ключа
  }
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// fetch с пределом ожидания на весь ответ, включая тело: read(res) читает его под тем же таймером,
// поэтому зависшая сеть и после заголовков даёт ошибку, а не бесконечный прогресс
async function fetchTimed(url, opts, ms, read) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), ms);
  try {
    const res = await fetch(url, { ...opts, signal: ctrl.signal });
    return await read(res);
  } finally {
    clearTimeout(timer);
  }
}
const readText = async (res) => ({ ok: res.ok, status: res.status, text: await res.text() });
const clamp01 = (x) => Math.min(1, Math.max(0, Number(x) || 0));
const reducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;

function mmss(sec) {
  const s = Math.max(0, Math.floor(Number(sec) || 0));
  return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
}
function mmssTenth(sec) {
  const t = Math.max(0, Number(sec) || 0);
  return `${mmss(t)}.${Math.floor((t * 10) % 10)}`;
}
const numRu = (x, digits) => Number(x).toLocaleString('ru-RU', { minimumFractionDigits: digits, maximumFractionDigits: digits });
const fmtSize = (bytes) => `${numRu(bytes / (1024 * 1024), 1)} МБ`;

const PLURAL = new Intl.PluralRules('ru-RU');
const VIOLATION_FORMS = { one: 'возможное нарушение', few: 'возможных нарушения', many: 'возможных нарушений', other: 'возможного нарушения' };
const violationsWord = (n) => `${n} ${VIOLATION_FORMS[PLURAL.select(n)]}`;

function confidenceWord(c) {
  if (c >= 0.8) return 'высокая';
  if (c >= 0.5) return 'средняя';
  return 'низкая';
}

function loadKey() {
  try { return localStorage.getItem(KEY_STORE) || ''; } catch { return ''; }
}
function saveKey(value) {
  try {
    if (value) localStorage.setItem(KEY_STORE, value);
    else localStorage.removeItem(KEY_STORE);
  } catch { /* приватный режим: ключ просто не запомнится */ }
}
/* Ключ из мессенджера или терминала часто приходит с невидимыми символами, кавычками или целой строкой
   «API_KEY=…»: убираем это. Заголовок HTTP принимает только латиницу — остальное ловим до отправки. */
function cleanKey(raw) {
  return (raw || '')
    .replace(/[\s\u00A0\u200B-\u200D\u2060\uFEFF]/g, '')
    .replace(/^API_KEY=/i, '')
    .replace(/^["'`«»“”‘’]+|["'`«»“”‘’]+$/g, '');
}
function keyProblem(key) {
  return /[^\x21-\x7E]/.test(key)
    ? 'Ключ скопирован с лишними символами (русская буква или кавычки). Скопируйте ключ заново и вставьте.'
    : '';
}
function authHeaders() {
  const key = cleanKey(el.key.value);
  return key ? { 'X-API-Key': key } : {};
}

function setJobInUrl(id) {
  const p = new URLSearchParams(location.search);
  if (id) p.set('job', id);
  else p.delete('job');
  const qs = p.toString();
  history.replaceState(null, '', qs ? `?${qs}` : location.pathname);
}

function httpError(status, body, phase) {
  let detail = '';
  try {
    const j = JSON.parse(body);
    if (typeof j.detail === 'string') detail = j.detail;
  } catch { /* тело не JSON — подробностей нет */ }
  if (status === 401 || status === 403) {
    return new UiError('Ключ не подошёл. Проверьте API-ключ и нажмите «Проверить видео» ещё раз.', { detail, retry: phase, field: 'key' });
  }
  if (status === 413) return new UiError('Файл больше 200 МБ — сервис его не примет. Выберите видео поменьше.', { detail, retry: 'none' });
  if (status === 400 || status === 415 || status === 422) {
    return new UiError('Сервис не принял файл: нужен видеофайл в формате .mov.', { detail, retry: 'none' });
  }
  if (status === 429) {
    return new UiError('Сервис сейчас занят другими проверками. Подождите несколько секунд и нажмите «Повторить».', { detail, retry: phase });
  }
  if (status === 404) {
    return new UiError('Сервис не нашёл эту проверку — возможно, его перезапустили. Загрузите видео ещё раз.', { detail, retry: 'upload' });
  }
  return new UiError(`Сервис ответил ошибкой (код ${status}). Подождите минуту и нажмите «Повторить».`, { detail, retry: phase });
}

/* ---------- экраны ---------- */

function setView(view) {
  el.upload.hidden = view !== 'idle';
  el.progress.hidden = view !== 'progress';
  el.failure.hidden = view !== 'failure';
  el.result.hidden = view !== 'result';
}

function showFormError(text) {
  el.formError.textContent = text;
  el.formError.hidden = false;
}
function hideFormError() {
  el.formError.hidden = true;
}

function setService(stateName, text) {
  el.service.dataset.state = stateName;
  el.serviceText.textContent = text;
}

async function checkHealth() {
  if (MOCK) {
    setService('mock', 'Демо-режим: ответ из mock_details.json, бэкенд не нужен');
    return;
  }
  try {
    const reply = await fetchTimed(`${API}/health`, { cache: 'no-store' }, HEALTH_TIMEOUT_MS, readText);
    if (!reply.ok) throw new Error(String(reply.status));
    const h = JSON.parse(reply.text);
    setService('ok', h.engine === 'mock' ? 'Сервис на связи · вместо модели пока заглушка' : 'Сервис на связи');
  } catch {
    setService('down', 'Нет связи с сервисом — проверка не запустится');
  }
}

/* ---------- выбор файла ---------- */

function pickFile(f) {
  if (!f) return;
  hideFormError();
  if (!/\.mov$/i.test(f.name)) {
    showFormError(`«${f.name}» — не .mov. Сервис принимает видео с камеры в формате .mov.`);
    return;
  }
  if (f.size > MAX_BYTES) {
    showFormError(`Файл весит ${fmtSize(f.size)} — больше 200 МБ, сервис его не примет.`);
    return;
  }
  state.file = f;
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  state.objectUrl = URL.createObjectURL(f);
  el.fileName.textContent = f.name;
  el.fileSize.textContent = fmtSize(f.size);
  el.fileRow.hidden = false;
  el.start.focus();
}

/* ---------- прогресс ---------- */

function setBar(frac) {
  const pct = Math.round(clamp01(frac) * 100);
  el.barFill.style.width = `${pct}%`;
  el.bar.setAttribute('aria-valuenow', String(pct));
  return pct;
}

function markStep(name) {
  const idx = name === 'done' ? STEPS.length : STEPS.indexOf(name);
  for (const li of el.steps.children) {
    const i = STEPS.indexOf(li.dataset.step);
    li.classList.toggle('is-done', idx >= 0 && i < idx);
    li.classList.toggle('is-active', i === idx);
  }
}

function setUploadProgress(frac) {
  const pct = setBar(clamp01(frac) * UPLOAD_SHARE);
  el.progressTitle.textContent = `${STAGE_TITLE.upload} — ${Math.round(clamp01(frac) * 100)} %`;
  markStep('upload');
  el.cancel.textContent = 'Отменить загрузку';
  return pct;
}

function setAnalysisProgress(progress, stage) {
  const pct = setBar(UPLOAD_SHARE + clamp01(progress) * (1 - UPLOAD_SHARE));
  el.progressTitle.textContent = `${STAGE_TITLE[stage] || 'Анализируем видео'} — ${pct} %`;
  markStep(STEPS.includes(stage) ? stage : 'decode');
  // API отмены нет: после загрузки кнопка только прекращает ожидание, сервер анализ доводит
  el.cancel.textContent = 'Прекратить ожидание';
}

function startClock() {
  if (!state.startedAt) state.startedAt = Date.now();
  clearInterval(state.clockTimer);
  const tick = () => {
    const sec = (Date.now() - state.startedAt) / 1000;
    el.elapsed.textContent = `прошло ${mmss(sec)}`;
    if (sec > 300 && !el.progressNote.textContent) {
      el.progressNote.textContent = 'Обработка идёт дольше 5 минут. Сервис ещё работает — можно подождать или прекратить ожидание.';
    }
  };
  tick();
  state.clockTimer = setInterval(tick, 1000);
}

function stopTimers() {
  clearTimeout(state.pollTimer);
  clearInterval(state.clockTimer);
}

/* ---------- запуск и опрос ---------- */

async function start() {
  if (!state.file && !MOCK) return;
  if (!MOCK && !cleanKey(el.key.value)) {
    showFormError('Введите API-ключ — без него сервис не примет видео.');
    el.key.focus();
    return;
  }
  if (!MOCK && keyProblem(cleanKey(el.key.value))) {
    showFormError(keyProblem(cleanKey(el.key.value)));
    el.key.focus();
    return;
  }
  hideFormError();
  const run = ++state.runId;
  state.startedAt = 0;
  state.netFails = 0;
  state.lastError = null;
  el.progressNote.textContent = '';
  setBar(0);
  setView('progress');
  startClock();
  setUploadProgress(0);
  try {
    if (MOCK) {
      await mockRun(run);
      return;
    }
    const jobId = await upload(state.file);
    if (run !== state.runId) return;
    state.jobId = jobId;
    setJobInUrl(jobId);
    setAnalysisProgress(0, 'queued');
    schedulePoll(0, run);
  } catch (err) {
    if (run === state.runId) fail(err);
  }
}

function upload(file) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    state.xhr = xhr;
    let stalled = false;
    let stallTimer = 0;
    // пока файл уходит, ждём движения не дольше UPLOAD_STALL_MS; после отправки ждёт уже общий xhr.timeout
    const armStall = () => {
      clearTimeout(stallTimer);
      stallTimer = setTimeout(() => {
        stalled = true;
        xhr.abort();
      }, UPLOAD_STALL_MS);
    };
    const settle = () => {
      clearTimeout(stallTimer);
      state.xhr = null;
    };
    xhr.open('POST', `${API}/analyze`);
    xhr.timeout = UPLOAD_TIMEOUT_MS;
    for (const [k, v] of Object.entries(authHeaders())) xhr.setRequestHeader(k, v);
    xhr.upload.onprogress = (e) => {
      armStall();
      if (e.lengthComputable) setUploadProgress(e.loaded / e.total);
    };
    xhr.upload.onload = () => clearTimeout(stallTimer);
    xhr.onload = () => {
      settle();
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const id = JSON.parse(xhr.responseText).job_id;
          if (!id) throw new Error('no job_id');
          resolve(String(id));
        } catch {
          reject(new UiError('Сервис принял файл, но не вернул номер проверки. Нажмите «Повторить».'));
        }
      } else {
        reject(httpError(xhr.status, xhr.responseText, 'upload'));
      }
    };
    xhr.onerror = () => {
      settle();
      reject(new UiError('Видео не загрузилось: нет связи с сервисом. Проверьте интернет и нажмите «Повторить».'));
    };
    xhr.ontimeout = () => {
      settle();
      reject(new UiError('Сервис слишком долго не отвечал на загрузку. Проверьте интернет и нажмите «Повторить».'));
    };
    xhr.onabort = () => {
      settle();
      reject(stalled
        ? new UiError('Загрузка зависла: файл перестал уходить на сервис. Проверьте интернет и нажмите «Повторить».')
        : new UiError('Загрузка отменена.', { retry: 'upload' }));
    };
    const form = new FormData();
    form.append('file', file, file.name);
    armStall();
    xhr.send(form);
  });
}

function schedulePoll(delay, run) {
  clearTimeout(state.pollTimer);
  state.pollTimer = setTimeout(() => pollOnce(run), delay);
}

async function pollOnce(run) {
  if (run !== state.runId) return;
  let reply;
  try {
    reply = await fetchTimed(`${API}/jobs/${encodeURIComponent(state.jobId)}/details`, {
      headers: authHeaders(), cache: 'no-store',
    }, POLL_TIMEOUT_MS, readText);
  } catch {
    if (run !== state.runId) return;
    state.netFails += 1;
    if (state.netFails <= NET_RETRIES) {
      el.progressNote.textContent = 'Связь с сервисом прервалась — пробуем ещё раз…';
      schedulePoll(POLL_MS, run);
      return;
    }
    fail(new UiError('Нет связи с сервисом. Проверьте интернет и нажмите «Повторить» — проверка продолжится с того же места.', { retry: 'poll' }));
    return;
  }
  if (run !== state.runId) return;
  state.netFails = 0;
  if (el.progressNote.textContent.startsWith('Связь')) el.progressNote.textContent = '';
  if (!reply.ok) {
    fail(httpError(reply.status, reply.text, 'poll'));
    return;
  }
  let details;
  try {
    details = JSON.parse(reply.text);
  } catch {
    fail(new UiError('Сервис прислал ответ, который не получилось прочитать. Нажмите «Повторить».', { retry: 'poll' }));
    return;
  }
  if (run !== state.runId) return;
  if (details.status === 'done' && details.analysis) {
    finish(details);
    return;
  }
  if (details.status === 'failed') {
    fail(new UiError('Сервис не смог обработать это видео.', { detail: details.error || '', retry: state.file ? 'upload' : 'none' }));
    return;
  }
  setAnalysisProgress(details.progress, details.stage);
  schedulePoll(POLL_MS, run);
}

async function mockRun(run) {
  for (let i = 1; i <= 10; i += 1) {
    await sleep(110);
    if (run !== state.runId) return;
    setUploadProgress(i / 10);
  }
  if (MOCK === 'fail') {
    throw new UiError('Сервис не смог обработать это видео.', { detail: 'демо-ошибка (?mock=fail)', retry: 'upload' });
  }
  const res = await fetch('mock_details.json', { cache: 'no-store' });
  if (!res.ok) throw new UiError('Для демо-режима нужен файл ui/mock_details.json — его нет рядом со страницей.', { retry: 'none' });
  const details = await res.json();
  for (const [p, stage] of [[0.15, 'decode'], [0.5, 'detect'], [0.8, 'verify'], [1, 'report']]) {
    await sleep(450);
    if (run !== state.runId) return;
    setAnalysisProgress(p, stage);
  }
  if (MOCK === 'clean') {
    details.analysis.verdict = 'no_violation';
    details.analysis.violations = [];
  }
  if (state.file) details.filename = state.file.name;
  finish(details);
}

function cancel() {
  state.runId += 1;
  stopTimers();
  if (state.xhr) state.xhr.abort();
  state.jobId = null;
  setJobInUrl(null);
  setView('idle');
}

function fail(err) {
  stopTimers();
  const e = err instanceof UiError ? err : new UiError('Что-то пошло не так. Нажмите «Повторить».', { detail: String(err && err.message || err) });
  state.lastError = e;
  if (e.field === 'key') {
    state.jobId = null;
    setJobInUrl(null);
    setView('idle');
    showFormError(e.message);
    el.key.focus();
    el.key.select();
    return;
  }
  el.failureText.textContent = e.message;
  el.failureDetail.textContent = e.detail ? `Ответ сервиса: ${e.detail}` : '';
  el.failureDetail.hidden = !e.detail;
  const canRetry = (e.retry === 'poll' && state.jobId) || (e.retry === 'upload' && (state.file || MOCK));
  el.retry.hidden = !canRetry;
  setView('failure');
  (canRetry ? el.retry : el.chooseOther).focus();
}

function retry() {
  const e = state.lastError;
  if (e && e.retry === 'poll' && state.jobId) {
    const run = ++state.runId;
    state.netFails = 0;
    setView('progress');
    startClock();
    schedulePoll(0, run);
    return;
  }
  start();
}

function resetToIdle() {
  state.runId += 1;
  stopTimers();
  state.jobId = null;
  setJobInUrl(null);
  el.player.pause();
  el.file.value = '';
  setView('idle');
  if (state.file) el.start.focus();
}

/* ---------- результат ---------- */

function finish(details) {
  stopTimers();
  setBar(1);
  markStep('done');
  const a = details.analysis;
  const violations = [...(a.violations || [])].sort((x, y) => x.timecode_sec - y.timecode_sec);
  state.duration = Number(a.duration_sec) || 0;
  renderVerdict(a, violations, details);
  renderPlayer(violations);
  renderCases(violations);
  setView('result');
  el.verdictTitle.focus({ preventScroll: true });
  el.result.scrollIntoView({ block: 'start', behavior: reducedMotion() ? 'auto' : 'smooth' });
}

function renderVerdict(a, violations, details) {
  const bad = a.verdict === 'violation';
  el.verdict.className = `verdict ${bad ? 'is-bad' : 'is-good'}`;
  el.verdictIcon.innerHTML = bad ? ICON_BAD : ICON_GOOD;
  el.verdictTitle.textContent = bad ? 'Обнаружены признаки возможных нарушений' : 'Нарушение не обнаружено';
  if (bad) {
    el.verdictText.textContent = violations.length
      ? `${violationsWord(violations.length)}. По каждому ниже — кадр, таймкод, описание и пункт нормы; проверьте фрагмент в видео.`
      : 'Сервис отметил нарушение, но не передал подробностей — просмотрите видео целиком.';
  } else {
    el.verdictText.textContent = 'Видимых признаков нарушений кассовой дисциплины в ролике не найдено.';
  }
  const meta = [];
  if (details.filename) meta.push(details.filename);
  if (a.duration_sec) meta.push(`длительность ${mmss(a.duration_sec)}`);
  if (a.processing_sec != null) meta.push(a.processing_sec < 1 ? 'проверено меньше чем за секунду' : `проверено за ${numRu(a.processing_sec, 0)} с`);
  el.verdictMeta.textContent = meta.join(' · ');
  el.badgeMock.hidden = !/mock/i.test(a.engine || '');
}

function renderPlayer(violations) {
  el.playerNote.hidden = true;
  if (state.objectUrl) {
    el.player.hidden = false;
    if (el.player.getAttribute('src') !== state.objectUrl) el.player.src = state.objectUrl;
  } else {
    el.player.hidden = true;
    if (state.jobId && !MOCK) loadServerVideo(state.jobId, state.runId);
  }

  for (const node of el.timeline.querySelectorAll('.tl-event')) node.remove();
  el.timelineLabel.textContent = violations.length
    ? 'Шкала ролика: окна событий отмечены, нажмите, чтобы перейти'
    : 'Шкала ролика: событий не отмечено';
  el.playhead.style.left = '0%';
  const dur = state.duration || el.player.duration || 0;
  el.ticks.replaceChildren(...[0, dur / 2, dur].map((t) => {
    const s = document.createElement('span');
    s.textContent = mmss(t);
    return s;
  }));
  if (!dur) return;

  violations.forEach((v, i) => {
    const [from, to] = eventWindow(v, dur);
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'tl-event';
    b.dataset.index = String(i);
    b.style.left = `${(from / dur) * 100}%`;
    b.style.width = `${Math.max(((to - from) / dur) * 100, 0.5)}%`;
    b.setAttribute('aria-label', `Событие ${i + 1}: ${v.type}, ${mmss(v.timecode_sec)}`);
    const num = document.createElement('span');
    num.textContent = String(i + 1);
    b.append(num);
    b.addEventListener('click', () => {
      seek(from, i);
      focusCase(i);
    });
    el.timeline.append(b);
  });
}

// страницу перезагрузили — локального файла нет, берём исходник с сервера (с ключом, поэтому через fetch, а не src)
async function loadServerVideo(jobId, run) {
  try {
    const blob = await fetchTimed(`${API}/jobs/${encodeURIComponent(jobId)}/video`, { headers: authHeaders(), cache: 'no-store' }, VIDEO_TIMEOUT_MS, (res) => {
      if (!res.ok) throw new Error(String(res.status));
      return res.blob();
    });
    if (run !== state.runId) return;
    if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
    state.objectUrl = URL.createObjectURL(blob);
    el.player.src = state.objectUrl;
    el.player.hidden = false;
  } catch {
    if (run !== state.runId) return;
    el.playerNote.textContent = 'Видео с сервера не загрузилось — кадры-доказательства в карточках ниже.';
    el.playerNote.hidden = false;
  }
}

function eventWindow(v, dur) {
  const t = Number(v.timecode_sec) || 0;
  const from = v.start_sec != null ? Number(v.start_sec) : Math.max(0, t - 1.5);
  const to = v.end_sec != null ? Number(v.end_sec) : Math.min(dur || t + 1.5, t + 1.5);
  return [Math.max(0, Math.min(from, t)), Math.max(to, t)];
}

function seek(sec, index) {
  for (const b of el.timeline.querySelectorAll('.tl-event')) b.classList.toggle('is-active', b.dataset.index === String(index));
  if (el.player.hidden || el.player.error) return;
  el.player.currentTime = Math.max(0, sec);
  el.player.play().catch(() => { /* автозапуск запрещён — пользователь нажмёт play сам */ });
}

function focusCase(i) {
  const node = el.cases.children[i];
  if (!node) return;
  for (const c of el.cases.children) c.classList.remove('is-target');
  node.classList.add('is-target');
  node.scrollIntoView({ block: 'start', behavior: reducedMotion() ? 'auto' : 'smooth' });
}

function frameSrc(b64) {
  return b64.startsWith('data:') ? b64 : `data:image/jpeg;base64,${b64}`;
}

function renderCases(violations) {
  el.cases.replaceChildren(...violations.map((v, i) => renderCase(v, i)));
}

function renderCase(v, i) {
  const node = el.tplCase.content.firstElementChild.cloneNode(true);
  node.style.setProperty('--i', String(i));
  node.id = `case-${i + 1}`;
  $('.case-num', node).textContent = String(i + 1);
  $('.case-type', node).textContent = v.type;

  const tc = $('.timecode', node);
  $('.timecode-text', tc).textContent = mmss(v.timecode_sec);
  tc.setAttribute('aria-label', `Перейти к ${mmss(v.timecode_sec)} в видео`);
  tc.addEventListener('click', () => {
    const [from] = eventWindow(v, state.duration);
    seek(from, i);
    el.playerPanel.scrollIntoView({ block: 'start', behavior: reducedMotion() ? 'auto' : 'smooth' });
  });

  const [from, to] = eventWindow(v, state.duration);
  $('.case-window .mono', node).textContent = `${mmss(from)}–${mmss(to)}`;
  const conf = $('.confidence', node);
  if (typeof v.confidence === 'number') {
    $('.confidence-text', conf).textContent = `уверенность: ${confidenceWord(v.confidence)} · ${numRu(v.confidence, 2)}`;
    $('.confidence-bar span', conf).style.width = `${Math.round(clamp01(v.confidence) * 100)}%`;
  } else {
    conf.remove();
  }

  $('.case-desc', node).textContent = v.description;
  const why = $('.case-why', node);
  if (v.why) $('span', why).textContent = v.why;
  else why.remove();

  const frames = $('.frames', node);
  const list = (v.evidence_frames || []).slice(0, 3);
  list.forEach((f, j) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'frame';
    const img = document.createElement('img');
    img.src = frameSrc(f.image_base64);
    img.alt = `Кадр-доказательство ${j + 1} из ${list.length}: ${v.type}, ${mmssTenth(f.timestamp_sec)}`;
    img.loading = 'lazy';
    img.decoding = 'async';
    const stamp = document.createElement('span');
    stamp.className = 'stamp';
    stamp.textContent = mmssTenth(f.timestamp_sec);
    btn.append(img, stamp);
    btn.setAttribute('aria-label', `Открыть крупно: ${img.alt}`);
    btn.addEventListener('click', () => openFrame(img.src, `${v.type} · кадр ${j + 1} из ${list.length} · ${mmssTenth(f.timestamp_sec)}`));
    frames.append(btn);
  });

  const rules = $('.rules', node);
  if (v.rules && v.rules.length) {
    rules.replaceChildren(...v.rules.map((r, k) => renderRule(r, v.rules[k - 1])));
  } else {
    const p = document.createElement('p');
    p.className = 'case-empty';
    p.textContent = 'Пункт нормы указан в описании выше.';
    rules.replaceChildren(p);
  }
  return node;
}

function renderRule(r, prev) {
  const node = el.tplRule.content.firstElementChild.cloneNode(true);
  const basis = $('.basis', node);
  const direct = r.basis === 'direct';
  basis.textContent = direct ? 'Прямая норма' : 'Трактовка';
  basis.classList.add(direct ? 'is-direct' : 'is-interp');
  $('.rule-ref', node).textContent = r.ref;
  $('.rule-approved', node).textContent = r.approved ? r.approved.charAt(0).toUpperCase() + r.approved.slice(1) : '';
  const act = $('.rule-act', node);
  act.textContent = r.act || '';
  act.addEventListener('click', () => act.setAttribute('aria-expanded', String(act.getAttribute('aria-expanded') !== 'true')));
  // второй пункт того же документа: название уже стоит выше
  if (prev && prev.act === r.act && prev.approved === r.approved) $('.rule-doc', node).remove();
  $('.rule-quote', node).textContent = r.quote ? `«${r.quote}»` : '';
  if (!r.quote) $('.rule-quote', node).remove();
  const link = $('.rule-link', node);
  if (/^https:\/\//.test(r.url || '')) link.href = r.url;
  else link.remove();
  return node;
}

function openFrame(src, caption) {
  el.lightboxImg.src = src;
  el.lightboxImg.alt = caption;
  el.lightboxCaption.textContent = caption;
  if (typeof el.lightbox.showModal === 'function') el.lightbox.showModal();
  else window.open(src, '_blank', 'noopener');
}

/* ---------- события ---------- */

el.key.value = loadKey();
el.key.addEventListener('input', () => {
  saveKey(cleanKey(el.key.value));
  hideFormError();
});
el.keyToggle.addEventListener('click', () => {
  const show = el.key.type === 'password';
  el.key.type = show ? 'text' : 'password';
  el.keyToggle.textContent = show ? 'Скрыть' : 'Показать';
});

el.file.addEventListener('change', () => pickFile(el.file.files[0]));
for (const type of ['dragenter', 'dragover']) {
  el.drop.addEventListener(type, (e) => {
    e.preventDefault();
    el.drop.classList.add('is-over');
  });
}
for (const type of ['dragleave', 'drop']) {
  el.drop.addEventListener(type, (e) => {
    e.preventDefault();
    el.drop.classList.remove('is-over');
  });
}
el.drop.addEventListener('drop', (e) => pickFile(e.dataTransfer && e.dataTransfer.files[0]));
// файл, брошенный мимо зоны, не должен открываться вместо страницы
window.addEventListener('dragover', (e) => e.preventDefault());
window.addEventListener('drop', (e) => e.preventDefault());

el.start.addEventListener('click', start);
el.demo.addEventListener('click', start);
el.cancel.addEventListener('click', cancel);
el.retry.addEventListener('click', retry);
el.chooseOther.addEventListener('click', () => {
  resetToIdle();
  el.file.click();
});
el.again.addEventListener('click', () => {
  state.file = null;
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  state.objectUrl = null;
  el.player.removeAttribute('src');
  el.player.load();
  el.fileRow.hidden = true;
  resetToIdle();
});

el.player.addEventListener('error', () => {
  if (!el.player.getAttribute('src')) return;
  el.player.hidden = true;
  el.playerNote.textContent = 'Браузер не может воспроизвести этот файл (часто это кодек HEVC). Кадры-доказательства — в карточках ниже.';
  el.playerNote.hidden = false;
});
el.player.addEventListener('timeupdate', () => {
  const dur = state.duration || el.player.duration;
  if (dur) el.playhead.style.left = `${Math.min(100, (el.player.currentTime / dur) * 100)}%`;
});

el.lightboxClose.addEventListener('click', () => el.lightbox.close());
el.lightbox.addEventListener('click', (e) => {
  if (e.target === el.lightbox) el.lightbox.close();
});

/* ---------- старт ---------- */

el.demo.hidden = !MOCK;
setView('idle');
checkHealth();
if (MOCK && params.has('run')) {
  start();
} else if (state.jobId && !MOCK) {
  // страницу перезагрузили во время проверки — продолжаем опрос той же задачи
  const run = ++state.runId;
  setView('progress');
  startClock();
  setAnalysisProgress(0, 'queued');
  schedulePoll(0, run);
}
