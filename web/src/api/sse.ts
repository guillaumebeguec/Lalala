import type { SearchEvent } from "./types";

/** Parses an SSE stream read via fetch + ReadableStream (EventSource does not support POST). */
export async function readSse(res: Response, onEvent: (e: SearchEvent) => void, signal?: AbortSignal): Promise<void> {
  if (!res.body) throw new Error("no body");
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  const flush = (chunk: string) => {
    let type = "message";
    let data = "";
    for (const line of chunk.split("\n")) {
      if (line.startsWith("event:")) type = line.slice(6).trim();
      else if (line.startsWith("data:")) data += line.slice(5).trim();
    }
    if (!data || type === "end") return;
    onEvent({ type, data: JSON.parse(data) } as SearchEvent);
  };
  while (true) {
    if (signal?.aborted) { await reader.cancel(); return; }
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i: number;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      flush(buf.slice(0, i));
      buf = buf.slice(i + 2);
    }
  }
  if (buf.trim()) flush(buf);
}
