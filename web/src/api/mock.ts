import type { SearchApi, SearchEvent, SearchInput } from "./types";
import memory from "../mocks/memory.json";
import cover from "../mocks/cover.json";
import noisy from "../mocks/noisy.json";

type Fixture = { events: SearchEvent[] };
const FIXTURES: Record<string, Fixture> = { memory: memory as Fixture, cover: cover as Fixture, noisy: noisy as Fixture };

const sleep = (ms: number, signal?: AbortSignal) =>
  new Promise<void>((resolve) => {
    const t = setTimeout(resolve, ms);
    signal?.addEventListener("abort", () => { clearTimeout(t); resolve(); }, { once: true });
  });

/** Rejoue les événements d'une fixture avec des délais réalistes (300–1 200 ms par étape). */
async function play(fx: Fixture, onEvent: (e: SearchEvent) => void, signal?: AbortSignal, fast = false) {
  for (const e of fx.events) {
    if (signal?.aborted) return;
    onEvent(e);
    if (fast) continue;
    if (e.type === "step" && e.data.status === "running") await sleep(300 + Math.random() * 900, signal);
    else if (e.type === "input") await sleep(250, signal);
  }
}

export function pickScene(input: SearchInput): string {
  if (input.video) return "cover";
  if (input.audio) return "noisy";
  return "memory";
}

export const mockApi: SearchApi = {
  async search(input, onEvent, signal) {
    const fx = FIXTURES[pickScene(input)];
    await play(fx, onEvent, signal, (globalThis as { __LALALA_FAST__?: boolean }).__LALALA_FAST__ === true);
  },
  async replay(scene, onEvent, signal) {
    await play(FIXTURES[scene] ?? FIXTURES.memory, onEvent, signal);
  },
  async feedback() {},
};
