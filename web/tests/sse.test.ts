import { describe, expect, it } from "vitest";
import { readSse } from "../src/api/sse";

function response(chunks: string[]): Response {
  const enc = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    start(c) { for (const ch of chunks) c.enqueue(enc.encode(ch)); c.close(); },
  });
  return new Response(stream);
}

describe("readSse", () => {
  it("parses events split across chunks and ignores end", async () => {
    const got: unknown[] = [];
    await readSse(response(["event: input\ndata: {\"kind\":\"te", "xt\",\"waveform\":[]}\n\nevent: step\ndata: {\"id\":\"a\",\"title\":\"A\",\"model\":\"m\",\"status\":\"done\",\"detail\":\"\"}\n\n", "event: end\ndata: {}\n\n"]), (e) => got.push(e));
    expect(got).toHaveLength(2);
    expect(got[0]).toMatchObject({ type: "input", data: { kind: "text" } });
    expect(got[1]).toMatchObject({ type: "step", data: { id: "a" } });
  });
});
