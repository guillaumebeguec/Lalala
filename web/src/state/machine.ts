import type { InputEvent, ResultEvent, SearchEvent, SearchInput, StepEvent, Track } from "../api/types";

// idle → capturing → searching → result | noMatch | error  (FRONT_PLAN.md § 5)
export type Phase = "idle" | "capturing" | "searching" | "result" | "noMatch" | "error";

export interface State {
  phase: Phase;
  input: SearchInput;
  inputInfo: InputEvent | null;
  steps: StepEvent[];
  result: ResultEvent | null;
  rejected: string[];
  error: string | null;
}

export type Action =
  | { type: "setInput"; input: Partial<SearchInput> }
  | { type: "clearInput" }
  | { type: "start" }
  | { type: "event"; event: SearchEvent }
  | { type: "fail"; message: string }
  | { type: "reject"; trackId: string }
  | { type: "promote"; trackId: string }
  | { type: "reset" }
  | { type: "backToCapture" };

export const initialState: State = { phase: "idle", input: {}, inputInfo: null, steps: [], result: null, rejected: [], error: null };

export function orderedCandidates(result: ResultEvent | null, rejected: string[]): Track[] {
  if (!result) return [];
  const all = [result.best, ...result.alternatives].filter((t): t is Track => !!t);
  return all.filter((t) => !rejected.includes(t.id));
}

export function reducer(s: State, a: Action): State {
  switch (a.type) {
    case "setInput":
      return { ...s, phase: "capturing", input: { ...s.input, ...a.input } };
    case "clearInput":
      return { ...s, phase: "idle", input: {} };
    case "start":
      return { ...s, phase: "searching", inputInfo: null, steps: [], result: null, rejected: [], error: null };
    case "event": {
      const e = a.event;
      if (e.type === "input") return { ...s, inputInfo: e.data };
      if (e.type === "step") {
        const i = s.steps.findIndex((x) => x.id === e.data.id);
        const steps = i >= 0 ? s.steps.map((x, k) => (k === i ? e.data : x)) : [...s.steps, e.data];
        return { ...s, steps };
      }
      if (e.type === "result") {
        const hasBest = !!e.data.best;
        return { ...s, result: e.data, phase: hasBest ? "result" : "noMatch" };   // low confidence → suggestions (top 10), no dead end
      }
      if (e.type === "error") return { ...s, phase: "error", error: e.data.message };
      return s;
    }
    case "fail":
      return { ...s, phase: "error", error: a.message };
    case "reject": {
      const rejected = [...s.rejected, a.trackId];
      const left = orderedCandidates(s.result, rejected);
      return { ...s, rejected, phase: left.length ? "result" : "noMatch" };
    }
    case "promote": {
      if (!s.result) return s;
      const all = orderedCandidates(s.result, s.rejected);
      const best = all.find((t) => t.id === a.trackId);
      if (!best) return s;
      return { ...s, result: { ...s.result, best, alternatives: all.filter((t) => t.id !== a.trackId) }, phase: "result" };
    }
    case "backToCapture":
      return { ...s, phase: Object.keys(s.input).length ? "capturing" : "idle", result: null, steps: [], error: null };
    case "reset":
      return initialState;
  }
}
