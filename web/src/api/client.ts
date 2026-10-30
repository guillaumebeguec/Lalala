import { readSse } from "./sse";
import type { ResultEvent, SearchApi, SearchEvent, SearchInput } from "./types";

function form(input: SearchInput): FormData {
  const fd = new FormData();
  if (input.text) fd.append("text", input.text);
  if (input.audio) fd.append("audio", input.audio, input.fileName ?? "recording.webm");
  if (input.video) fd.append("video", input.video, input.fileName ?? "clip.mp4");
  return fd;
}

export const liveApi: SearchApi = {
  async search(input, onEvent, signal) {
    try {
      const res = await fetch("/api/search/stream", { method: "POST", body: form(input), signal });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      await readSse(res, onEvent, signal);
    } catch (e) {
      if (signal?.aborted) return;
      // Repli : réponse JSON finale seule.
      const res = await fetch("/api/search", { method: "POST", body: form(input), signal });
      if (!res.ok) throw e instanceof Error ? e : new Error(String(e));
      const data = (await res.json()) as ResultEvent;
      onEvent({ type: "result", data });
    }
  },
  async replay(scene, onEvent, signal) {
    const res = await fetch(`/api/replay/${encodeURIComponent(scene)}`, { signal });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    await readSse(res, onEvent as (e: SearchEvent) => void, signal);
  },
  async feedback(payload) {
    await fetch("/api/feedback", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
  },
};
