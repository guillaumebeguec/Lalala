/** Enregistrement micro 10 s avec niveau en direct (MediaRecorder + AnalyserNode). */
export interface Recording { blob: Blob; durationS: number; mime: string }

export function pickMime(): string {
  const cands = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"];
  return cands.find((m) => typeof MediaRecorder !== "undefined" && MediaRecorder.isTypeSupported(m)) ?? "";
}

export async function record(opts: { maxS?: number; onLevel?: (v: number) => void; onTick?: (s: number) => void; signal?: AbortSignal }): Promise<Recording> {
  const maxS = opts.maxS ?? 10;
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const mime = pickMime();
  const rec = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
  const chunks: Blob[] = [];
  rec.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
  let ctx: AudioContext | null = null;
  let raf = 0;
  try {
    ctx = new AudioContext();
    const src = ctx.createMediaStreamSource(stream);
    const an = ctx.createAnalyser();
    an.fftSize = 512;
    src.connect(an);
    const buf = new Uint8Array(an.fftSize);
    const loop = () => {
      an.getByteTimeDomainData(buf);
      let sum = 0;
      for (const v of buf) { const d = (v - 128) / 128; sum += d * d; }
      opts.onLevel?.(Math.min(1, Math.sqrt(sum / buf.length) * 4));
      raf = requestAnimationFrame(loop);
    };
    loop();
  } catch { /* niveau non disponible : on enregistre quand même */ }
  const started = performance.now();
  const done = new Promise<Recording>((resolve) => {
    rec.onstop = () => {
      cancelAnimationFrame(raf);
      stream.getTracks().forEach((t) => t.stop());
      void ctx?.close();
      const blob = new Blob(chunks, { type: rec.mimeType || mime || "audio/webm" });
      resolve({ blob, durationS: Math.round((performance.now() - started) / 1000), mime: blob.type });
    };
  });
  rec.start(250);
  let s = 0;
  const tick = setInterval(() => { s += 1; opts.onTick?.(s); if (s >= maxS && rec.state === "recording") rec.stop(); }, 1000);
  opts.signal?.addEventListener("abort", () => { if (rec.state === "recording") rec.stop(); }, { once: true });
  const out = await done;
  clearInterval(tick);
  return out;
}

export function extensionFor(mime: string): string {
  return mime.includes("mp4") ? "m4a" : mime.includes("ogg") ? "ogg" : "webm";
}
