/* ── Alert chime — Web Audio API ────────────────────────────
   Plays a short 3-note ascending chime (C5 → E5 → G5).
   Total duration ≈ 1.2 s, well within the 2–3 s target.
   Checks localStorage 'soundEnabled' before playing so the
   check is always fresh (no stale React state).
   ─────────────────────────────────────────────────────────── */

export function playAlertChime() {
  if (localStorage.getItem('soundEnabled') === 'false') return;

  try {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) return;

    const ctx = new AudioCtx();

    /* Three notes: C5, E5, G5 */
    const notes = [
      { freq: 523.25, start: 0.0,  duration: 0.35 },
      { freq: 659.25, start: 0.38, duration: 0.35 },
      { freq: 783.99, start: 0.76, duration: 0.55 },
    ];

    notes.forEach(({ freq, start, duration }) => {
      const osc  = ctx.createOscillator();
      const gain = ctx.createGain();

      osc.connect(gain);
      gain.connect(ctx.destination);

      osc.type = 'sine';
      osc.frequency.value = freq;

      const t0 = ctx.currentTime + start;
      gain.gain.setValueAtTime(0, t0);
      gain.gain.linearRampToValueAtTime(0.28, t0 + 0.025);
      gain.gain.exponentialRampToValueAtTime(0.001, t0 + duration);

      osc.start(t0);
      osc.stop(t0 + duration + 0.05);
    });

    /* Release the audio context after all notes finish */
    setTimeout(() => {
      ctx.close().catch(() => {});
    }, 2000);
  } catch (_) {
    /* Web Audio API unavailable — silently skip */
  }
}

export function isSoundEnabled() {
  return localStorage.getItem('soundEnabled') !== 'false';
}

export function setSoundEnabled(enabled) {
  localStorage.setItem('soundEnabled', enabled ? 'true' : 'false');
}
