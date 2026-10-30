import { createContext, useCallback, useContext, useMemo, useReducer, useRef } from "react";
import { api } from "../api";
import type { SearchInput } from "../api/types";
import { initialState, reducer, type Action, type State } from "./machine";

interface Ctx { state: State; dispatch: (a: Action) => void; start: (input?: SearchInput) => Promise<void>; replay: (scene: string) => Promise<void>; cancel: () => void }

const SearchCtx = createContext<Ctx | null>(null);

export function SearchProvider({ children }: { children: React.ReactNode }) {
  const [state, dispatch] = useReducer(reducer, initialState);
  const abort = useRef<AbortController | null>(null);
  const cancel = useCallback(() => { abort.current?.abort(); abort.current = null; }, []);
  const run = useCallback(async (fn: (onEvent: Parameters<typeof api.search>[1], signal: AbortSignal) => Promise<void>) => {
    cancel();
    const ac = new AbortController();
    abort.current = ac;
    dispatch({ type: "start" });
    try {
      await fn((e) => { if (!ac.signal.aborted) dispatch({ type: "event", event: e }); }, ac.signal);
    } catch (e) {
      if (!ac.signal.aborted) dispatch({ type: "fail", message: e instanceof Error ? e.message : String(e) });
    }
  }, [cancel]);
  const start = useCallback(async (input?: SearchInput) => {
    const inp = input ?? state.input;
    if (input) dispatch({ type: "setInput", input });
    await run((onEvent, signal) => api.search(inp, onEvent, signal));
  }, [run, state.input]);
  const replay = useCallback(async (scene: string) => run((onEvent, signal) => api.replay(scene, onEvent, signal)), [run]);
  const value = useMemo(() => ({ state, dispatch, start, replay, cancel }), [state, start, replay, cancel]);
  return <SearchCtx.Provider value={value}>{children}</SearchCtx.Provider>;
}

export function useSearch(): Ctx {
  const c = useContext(SearchCtx);
  if (!c) throw new Error("SearchProvider manquant");
  return c;
}
