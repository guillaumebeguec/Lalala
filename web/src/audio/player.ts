/** Only one active player at a time across the whole app. */
let current: HTMLAudioElement | null = null;
const listeners = new Set<(el: HTMLAudioElement | null) => void>();

export function play(el: HTMLAudioElement) {
  if (current && current !== el) current.pause();
  current = el;
  listeners.forEach((l) => l(el));
  void el.play().catch(() => { /* autoplay refused: the user will click */ });
}

export function pause(el: HTMLAudioElement) {
  el.pause();
  if (current === el) { current = null; listeners.forEach((l) => l(null)); }
}

export function onActive(l: (el: HTMLAudioElement | null) => void): () => void {
  listeners.add(l);
  return () => listeners.delete(l);
}

export function fmt(s: number): string {
  if (!isFinite(s) || s < 0) s = 0;
  const m = Math.floor(s / 60), r = Math.floor(s % 60);
  return `${m}:${r.toString().padStart(2, "0")}`;
}
