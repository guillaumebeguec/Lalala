import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api";
import { Banner } from "../components/Banner";
import { CandidateRow } from "../components/CandidateRow";
import { CompareCards } from "../components/CompareCards";
import { Header } from "../components/Header";
import { ProofList } from "../components/ProofList";
import { ScoreDrawer } from "../components/ScoreDrawer";
import { SectionLabel } from "../components/SectionLabel";
import { TrackHero } from "../components/TrackHero";
import { copy } from "../copy";
import { orderedCandidates } from "../state/machine";
import { useSearch } from "../state/useSearch";

export function Result() {
  const { state, dispatch, cancel } = useSearch();
  const nav = useNavigate();
  const [drawer, setDrawer] = useState(false);
  const [yoursUrl, setYoursUrl] = useState<string | null>(null);
  useEffect(() => { if (!state.result) nav("/", { replace: true }); }, [state.result, nav]);
  useEffect(() => {
    const blob = state.input.audio ?? state.input.video;
    if (!blob) { setYoursUrl(null); return; }
    const u = URL.createObjectURL(blob); setYoursUrl(u);
    return () => URL.revokeObjectURL(u);
  }, [state.input.audio, state.input.video]);
  const candidates = useMemo(() => orderedCandidates(state.result, state.rejected), [state.result, state.rejected]);
  if (!state.result) return null;
  const best = state.phase === "result" ? candidates[0] : null;
  const others = best ? candidates.slice(1, 4) : candidates.slice(0, 4);
  const isAudio = state.inputInfo ? state.inputInfo.kind !== "text" : !!(state.input.audio || state.input.video);
  const newSearch = () => { cancel(); dispatch({ type: "reset" }); nav("/"); };
  const verdict = (v: "match" | "reject") => {
    void api.feedback({ query_id: state.result!.query_id, track_id: best?.id ?? null, verdict: v });
    if (v === "reject" && best) dispatch({ type: "reject", trackId: best.id });
    if (v === "match") newSearch();
  };
  return (
    <div className="mx-auto min-h-screen max-w-content px-6 py-6 md:px-10">
      <Header action={{ label: copy.result.newSearch, onClick: newSearch }} />
      {state.phase === "error" && <Banner onRetry={newSearch} />}
      <main className="grid gap-10 py-8 md:grid-cols-[1.4fr_1fr]">
        <section className="flex flex-col gap-8">
          <AnimatePresence mode="wait">
            {best ? (
              <motion.div key={best.id} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex flex-col gap-8">
                <TrackHero track={best} confidence={state.result.confidence} autoPlay />
                <ProofList proofs={state.result.proofs} />
                {isAudio && <CompareCards yours={yoursUrl} original={best.preview_url} />}
                <div className="flex flex-wrap items-center gap-3">
                  <button type="button" onClick={() => verdict("match")} className="min-h-[48px] rounded-pill bg-yellow px-6 text-[16px] font-bold text-on-accent">{copy.result.thatsTheOne}</button>
                  <button type="button" onClick={() => verdict("reject")} className="min-h-[48px] rounded-pill border border-border-strong px-6 text-[16px] font-medium text-text">{copy.result.notThisOne}</button>
                  <button type="button" onClick={() => setDrawer(true)} className="min-h-[44px] px-2 text-[15px] text-source underline">{copy.result.scoreDetails}</button>
                </div>
              </motion.div>
            ) : (
              <motion.div key="nomatch" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                <h1 className="font-display text-[40px] font-extrabold leading-[1.05] tracking-[-0.04em] md:text-[54px]">{copy.result.noMatchH1}</h1>
                <p className="mt-2 text-[17px] text-muted">{copy.result.noMatchSub}</p>
                <button type="button" onClick={() => { dispatch({ type: "backToCapture" }); nav("/"); }} className="mt-6 min-h-[48px] rounded-pill bg-yellow px-6 text-[16px] font-bold text-on-accent">{copy.result.addDetail}</button>
              </motion.div>
            )}
          </AnimatePresence>
        </section>
        <aside>
          <SectionLabel>{copy.result.others}</SectionLabel>
          <ul className="flex flex-col gap-3">
            {others.map((t) => <CandidateRow key={t.id} track={t} onPromote={(id) => dispatch({ type: "promote", trackId: id })} />)}
          </ul>
        </aside>
      </main>
      <ScoreDrawer open={drawer} onClose={() => setDrawer(false)} result={state.result} tracks={[...candidates]} />
    </div>
  );
}
