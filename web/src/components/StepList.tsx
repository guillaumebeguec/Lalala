import { motion } from "framer-motion";
import type { StepEvent } from "../api/types";
import { copy } from "../copy";
import { SectionLabel } from "./SectionLabel";

function Dot({ status }: { status: StepEvent["status"] }) {
  const base = "flex h-7 w-7 shrink-0 items-center justify-center rounded-full font-mono text-[13px]";
  if (status === "done") return <motion.span initial={{ scale: 0.6 }} animate={{ scale: 1 }} transition={{ type: "spring", stiffness: 500, damping: 18 }} className={`${base} bg-yellow text-on-accent`}>✓</motion.span>;
  if (status === "running") return <span className={`${base} bg-accent text-on-accent`}>♪</span>;
  if (status === "skipped") return <span className={`${base} bg-chip text-dim`}>–</span>;
  return <span className={`${base} bg-chip text-dim`}>·</span>;
}

export function StepList({ steps }: { steps: StepEvent[] }) {
  return (
    <section>
      <SectionLabel>{copy.searching.progress}</SectionLabel>
      <ol className="flex flex-col gap-2" aria-live="polite">
        {steps.map((s) => (
          <motion.li key={s.id} layout animate={{ backgroundColor: s.status === "running" ? "#211C3D" : "rgba(33,28,61,0)" }}
            className="flex gap-4 rounded-row px-4 py-3">
            <Dot status={s.status} />
            <div className="min-w-0">
              <div className={`text-[16px] font-medium ${s.status === "pending" ? "text-dim" : s.status === "skipped" ? "text-dim line-through" : "text-text"}`}>{s.title}</div>
              <div className="font-mono text-[12px] text-dim">{s.model}</div>
              {(s.detail || s.status === "skipped") && <div className="mt-1 text-[14px] text-muted">{s.status === "skipped" ? copy.searching.skipped : s.detail}</div>}
            </div>
          </motion.li>
        ))}
      </ol>
    </section>
  );
}
