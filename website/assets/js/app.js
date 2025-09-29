/* Client-side visualization logic (enhanced) */
(function () {
  const audioEl = document.getElementById('audio');
  const audioInput = document.getElementById('audioInput');
  const jsonInput = document.getElementById('jsonInput');
  const canvas = document.getElementById('waveCanvas');
  const ctx = canvas.getContext('2d');
  const segmentsList = document.getElementById('segmentsList');
  const dropzone = document.getElementById('dropzone');
  const yearEl = document.getElementById('year');

  if (yearEl) yearEl.textContent = new Date().getFullYear().toString();

  let audioBuffer = null; // AudioBuffer from Web Audio API
  let objectUrl = null;   // blob URL for <audio>
  let segments = [];      // [{start, end}]
  let rafId = null;       // requestAnimationFrame id for playhead

  // Drag & drop
  ;['dragenter','dragover'].forEach(evt => dropzone.addEventListener(evt, (e) => { e.preventDefault(); e.stopPropagation(); dropzone.classList.add('dragover'); }));
  ;['dragleave','drop'].forEach(evt => dropzone.addEventListener(evt, (e) => { e.preventDefault(); e.stopPropagation(); dropzone.classList.remove('dragover'); }));
  dropzone.addEventListener('drop', async (e) => {
    const files = [...(e.dataTransfer?.files || [])];
    if (!files.length) return;
    const audioFile = files.find(f => f.type.startsWith('audio/')) || files.find(f => /\.(wav|mp3|m4a|ogg)$/i.test(f.name));
    const jsonFile = files.find(f => f.type === 'application/json') || files.find(f => /segments\.json$/i.test(f.name));
    if (audioFile) await loadAudioFile(audioFile);
    if (jsonFile) await loadJsonFile(jsonFile);
  });

  // Resize handling to redraw waveform responsively
  const resizeObserver = new ResizeObserver(() => { if (audioBuffer) drawWaveform(audioBuffer, segments); });
  resizeObserver.observe(canvas.parentElement);
  window.addEventListener('resize', () => { if (audioBuffer) drawWaveform(audioBuffer, segments); });

  audioInput.addEventListener('change', async (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    await loadAudioFile(file);
  });

  jsonInput.addEventListener('change', async (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    await loadJsonFile(file);
  });

  async function loadAudioFile(file) {
    // Set audio element source
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = URL.createObjectURL(file);
    audioEl.src = objectUrl;

    // Decode for visualization
    try {
      const arrayBuffer = await file.arrayBuffer();
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
      drawWaveform(audioBuffer, segments);
    } catch (err) {
      console.error('Failed to decode audio:', err);
      alert('Failed to decode audio for visualization. Try a different file.');
    }
  }

  async function loadJsonFile(file) {
    try {
      const text = await file.text();
      const data = JSON.parse(text);
      if (!Array.isArray(data)) throw new Error('Invalid segments.json format');
      segments = data.map((d) => ({ start: Number(d.start), end: Number(d.end) }))
                     .filter((d) => Number.isFinite(d.start) && Number.isFinite(d.end) && d.end > d.start);
      renderSegmentsList(segments);
      if (audioBuffer) drawWaveform(audioBuffer, segments);
    } catch (err) {
      console.error('Failed to read segments.json:', err);
      alert('Could not read segments.json. Ensure it is an array of {start, end} in seconds.');
    }
  }

  function renderSegmentsList(list) {
    segmentsList.innerHTML = '';
    segmentsList.classList.toggle('empty', list.length === 0);
    if (list.length === 0) {
      segmentsList.textContent = 'No segments loaded.';
      return;
    }

    list.forEach((seg, idx) => {
      const item = document.createElement('div');
      item.className = 'segment-item';

      const meta = document.createElement('div');
      meta.className = 'segment-meta';
      const dur = Math.max(0, (seg.end - seg.start));
      meta.textContent = `#${String(idx + 1).padStart(2, '0')}  ` +
        `start: ${seg.start.toFixed(2)}s, end: ${seg.end.toFixed(2)}s`;

      const badge = document.createElement('span');
      badge.className = 'badge';
      badge.textContent = `${dur.toFixed(2)}s`;
      meta.appendChild(badge);

      const actions = document.createElement('div');
      actions.className = 'segment-actions';
      const playBtn = document.createElement('button');
      playBtn.textContent = 'Play segment';
      playBtn.addEventListener('click', () => playSegment(seg.start, seg.end));
      actions.appendChild(playBtn);

      item.appendChild(meta);
      item.appendChild(actions);
      segmentsList.appendChild(item);
    });
  }

  function drawWaveform(buffer, segs) {
    const parent = canvas.parentElement;
    const dpr = window.devicePixelRatio || 1;
    const widthCss = parent.clientWidth || 1000;
    const heightCss = 260; // a bit taller
    canvas.width = Math.floor(widthCss * dpr);
    canvas.height = Math.floor(heightCss * dpr);
    canvas.style.width = widthCss + 'px';
    canvas.style.height = heightCss + 'px';

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, widthCss, heightCss);

    const chData = buffer.getChannelData(0);
    const length = chData.length;
    const step = Math.max(1, Math.ceil(length / widthCss));
    const amp = heightCss / 2;

    // Background
    ctx.fillStyle = '#0b1220';
    ctx.fillRect(0, 0, widthCss, heightCss);

    // Midline
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.beginPath();
    ctx.moveTo(0, amp);
    ctx.lineTo(widthCss, amp);
    ctx.stroke();

    // Waveform min/max bars
    ctx.strokeStyle = '#60a5fa';
    ctx.beginPath();
    for (let x = 0; x < widthCss; x++) {
      const start = x * step;
      let min = 1.0;
      let max = -1.0;
      for (let i = 0; i < step && (start + i) < length; i++) {
        const v = chData[start + i];
        if (v < min) min = v;
        if (v > max) max = v;
      }
      const y1 = (1 - max) * amp;
      const y2 = (1 - min) * amp;
      ctx.moveTo(x, y1);
      ctx.lineTo(x, y2);
    }
    ctx.stroke();

    // Segments overlay
    if (Array.isArray(segs) && segs.length > 0 && Number.isFinite(buffer.duration)) {
      ctx.fillStyle = 'rgba(52, 211, 153, 0.22)';
      ctx.strokeStyle = 'rgba(52, 211, 153, 0.75)';
      ctx.lineWidth = 1.5;
      segs.forEach((seg) => {
        const x1 = (seg.start / buffer.duration) * widthCss;
        const x2 = (seg.end / buffer.duration) * widthCss;
        const w = Math.max(1, x2 - x1);
        ctx.fillRect(x1, 0, w, heightCss);
        ctx.strokeRect(x1 + 0.5, 0.5, Math.max(0, w - 1), heightCss - 1);
      });
    }

    // Playhead label element
    ensurePlayheadLabel();
    updatePlayhead();
  }

  function ensurePlayheadLabel() {
    const container = canvas.parentElement;
    if (!container.querySelector('.playhead-label')) {
      const div = document.createElement('div');
      div.className = 'playhead-label';
      div.textContent = '0.00s';
      container.appendChild(div);
    }
  }

  function updatePlayhead() {
    cancelAnimationFrame(rafId);
    rafId = requestAnimationFrame(drawPlayhead);
  }

  function drawPlayhead() {
    if (!audioBuffer) return;
    const container = canvas.parentElement;
    const label = container.querySelector('.playhead-label');

    // Redraw overlay playhead line without recomputing waveform (clear minimal area)
    const dpr = window.devicePixelRatio || 1;
    const widthCss = canvas.clientWidth;
    const heightCss = canvas.clientHeight;
    const width = canvas.width / dpr;
    const height = canvas.height / dpr;

    // Clear top overlay by redrawing waveform is heavy; instead, re-render everything when needed.
    // For simplicity and visual quality, trigger full redraw at ~60fps only while playing.
    if (!audioEl.paused && !audioEl.ended) {
      drawWaveform(audioBuffer, segments);
    }

    // Draw playhead on top
    const progress = Math.max(0, Math.min(1, (audioEl.currentTime || 0) / (audioBuffer.duration || 1)));
    const x = Math.floor(progress * width);
    const ctx2 = ctx;
    ctx2.save();
    ctx2.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx2.strokeStyle = 'rgba(96,165,250,0.95)';
    ctx2.lineWidth = 1.5;
    ctx2.beginPath();
    ctx2.moveTo(x + 0.5, 0);
    ctx2.lineTo(x + 0.5, height);
    ctx2.stroke();
    ctx2.restore();

    if (label) label.textContent = `${(audioEl.currentTime || 0).toFixed(2)}s`;

    if (!audioEl.paused && !audioEl.ended) {
      rafId = requestAnimationFrame(drawPlayhead);
    }
  }

  function playSegment(startSec, endSec) {
    if (!Number.isFinite(startSec) || !Number.isFinite(endSec) || endSec <= startSec) return;

    const onTimeUpdate = () => {
      if (audioEl.currentTime >= endSec) {
        audioEl.pause();
        audioEl.removeEventListener('timeupdate', onTimeUpdate);
        updatePlayhead();
      }
    };

    // Ensure metadata is loaded before seeking
    if (Number.isNaN(audioEl.duration)) {
      audioEl.addEventListener('loadedmetadata', () => {
        audioEl.currentTime = Math.min(startSec, audioEl.duration || startSec);
        audioEl.play();
        updatePlayhead();
      }, { once: true });
    } else {
      audioEl.currentTime = Math.min(startSec, audioEl.duration || startSec);
      audioEl.play();
      updatePlayhead();
    }

    audioEl.addEventListener('timeupdate', onTimeUpdate);
  }

  audioEl.addEventListener('play', updatePlayhead);
  audioEl.addEventListener('pause', updatePlayhead);
  audioEl.addEventListener('seeked', updatePlayhead);
})();
