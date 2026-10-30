import { AnimatePresence, motion } from "framer-motion";
import type { ResultEvent, Track } from "../api/types";
import { copy } from "../copy";

export function ScoreDrawer({ open, onClose, result, tracks }: { open: boolean; onClose: () => void; result: ResultEvent; tracks: Track[] }) {
  const name = (id: string) => tracks.find((t) => t.id === id)?.title ?? id;
  const embNames = Array.from(new Set(result.scores.flatMap((r) => Object.keys(r.emb ?? {}))));
  return (
    <AnimatePresence>
      {open && (
        <motion.aside initial={{ x: "100%" }} animate={{ x: 0 }} exit={{ x: "100%" }} transition={{ type: "tween", duration: 0.25 }}
          className="fixed inset-y-0 right-0 z-20 w-full max-w-[560px] overflow-y-auto border-l border-border bg-surface p-6 shadow-2xl" role="dialog" aria-label={copy.result.drawerTitle}>
          <div className="mb-4 flex items-center justify-between">
            <h2 className="font-display text-[26px] font-extrabold tracking-[-0.03em]">{copy.result.drawerTitle}</h2>
            <button type="button" onClick={onClose} className="rounded-pill border border-border-strong px-4 py-2 text-[14px]">{copy.result.close}</button>
          </div>
          <h3 className="mb-2 font-mono text-[12px] uppercase tracking-[0.1em] text-muted">{copy.result.drawerScores}</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-left font-mono text-[12px]">
              <thead className="text-dim"><tr><th className="py-1 pr-3">Track</th>{result.scores.some((r) => r.bm25 != null) && <th className="pr-3">BM25</th>}{embNames.map((e) => <th key={e} className="pr-3">{e}</th>)}<th className="pr-3">Rerank</th><th>Final</th></tr></thead>
              <tbody>
                {result.scores.map((r) => (
                  <tr key={r.track_id} className="border-t border-border text-text-2">
                    <td className="py-1 pr-3 text-text">{name(r.track_id)}</td>
                    {result.scores.some((x) => x.bm25 != null) && <td className="pr-3">{r.bm25?.toFixed(2) ?? "–"}</td>}
                    {embNames.map((e) => <td key={e} className="pr-3">{r.emb?.[e] ? `${r.emb[e].score.toFixed(2)} #${r.emb[e].rank}` : "–"}</td>)}
                    <td className="pr-3">{r.rerank ?? "–"}</td>
                    <td className="text-yellow">{r.final.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {result.timings.length > 0 && (
            <>
              <h3 className="mb-2 mt-6 font-mono text-[12px] uppercase tracking-[0.1em] text-muted">{copy.result.drawerTimings}</h3>
              <ul className="font-mono text-[13px] text-text-2">{result.timings.map((t) => <li key={t.step} className="flex justify-between border-t border-border py-1"><span>{t.step}</span><span>{t.ms} ms</span></li>)}</ul>
            </>
          )}
          {Object.keys(result.analysis ?? {}).length > 0 && (
            <>
              <h3 className="mb-2 mt-6 font-mono text-[12px] uppercase tracking-[0.1em] text-muted">{copy.result.drawerAnalysis}</h3>
              <pre className="whitespace-pre-wrap rounded-row bg-bg p-3 font-mono text-[12px] text-text-2">{JSON.stringify(result.analysis, null, 1)}</pre>
            </>
          )}
        </motion.aside>
      )}
    </AnimatePresence>
  );
}
