import { useEffect } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { Header } from "../components/Header";
import { InputCard } from "../components/InputCard";
import { StepList } from "../components/StepList";
import { ThreeDots } from "../components/ThreeDots";
import { Banner } from "../components/Banner";
import { copy } from "../copy";
import { useSearch } from "../state/useSearch";

export function Searching() {
  const { state, cancel, start, dispatch } = useSearch();
  const nav = useNavigate();
  const { id } = useParams();
  useEffect(() => { if (state.phase === "result" || state.phase === "noMatch") nav(`/result/${id}`, { replace: true }); }, [state.phase, id, nav]);
  useEffect(() => { if (state.phase === "idle") nav("/", { replace: true }); }, [state.phase, nav]);
  const done = state.steps.filter((s) => s.status === "done" || s.status === "skipped").length;
  const analysed = state.steps.length ? Math.min(1, (done + 0.5) / state.steps.length) : 0.2;
  return (
    <div className="mx-auto min-h-screen max-w-content px-6 py-6 md:px-10">
      <Header action={{ label: copy.searching.cancel, onClick: () => { cancel(); dispatch({ type: "backToCapture" }); nav("/"); } }} />
      {state.phase === "error" && <Banner onRetry={() => void start()} />}
      <main className="grid gap-10 py-8 md:grid-cols-[1.1fr_1fr]">
        <section className="flex flex-col gap-6">
          <ThreeDots />
          <div>
            <h1 className="font-display text-[36px] font-extrabold leading-[1.05] tracking-[-0.04em] md:text-[46px]">{copy.searching.h1}</h1>
            <p className="mt-2 text-[16px] text-muted">{copy.searching.sub}</p>
          </div>
          <InputCard info={state.inputInfo} analysed={analysed} />
        </section>
        <StepList steps={state.steps} />
      </main>
    </div>
  );
}
