import { describe, expect, it } from "vitest";
import { initialState, orderedCandidates, reducer } from "../src/state/machine";
import memory from "../src/mocks/memory.json";
import type { SearchEvent } from "../src/api/types";

const events = (memory as { events: SearchEvent[] }).events;

describe("machine", () => {
  it("replays a fixture to the result phase", () => {
    let s = reducer(initialState, { type: "start" });
    for (const e of events) s = reducer(s, { type: "event", event: e });
    expect(s.phase).toBe("result");
    expect(s.steps.every((x) => x.status === "done" || x.status === "skipped")).toBe(true);
    expect(s.inputInfo?.kind).toBe("text");
    expect(orderedCandidates(s.result, s.rejected).map((t) => t.id)).toEqual(["mock1", "mock2", "mock3", "mock4"]);
  });
  it("rejects, promotes and falls to noMatch", () => {
    let s = reducer(initialState, { type: "start" });
    for (const e of events) s = reducer(s, { type: "event", event: e });
    s = reducer(s, { type: "reject", trackId: "mock1" });
    expect(s.phase).toBe("result");
    expect(orderedCandidates(s.result, s.rejected)[0].id).toBe("mock2");
    s = reducer(s, { type: "promote", trackId: "mock4" });
    expect(s.result?.best?.id).toBe("mock4");
    for (const id of ["mock2", "mock3", "mock4"]) s = reducer(s, { type: "reject", trackId: id });
    expect(s.phase).toBe("noMatch");
  });
  it("keeps the input when going back to capture", () => {
    let s = reducer(initialState, { type: "setInput", input: { text: "whistling" } });
    s = reducer(s, { type: "start" });
    s = reducer(s, { type: "fail", message: "boom" });
    expect(s.phase).toBe("error");
    s = reducer(s, { type: "backToCapture" });
    expect(s.phase).toBe("capturing");
    expect(s.input.text).toBe("whistling");
  });
});
