import { motion } from "framer-motion";
import { copy } from "../copy";
import { SectionLabel } from "./SectionLabel";

export function ProofList({ proofs }: { proofs: { source: string; text: string }[] }) {
  if (!proofs.length) return null;
  return (
    <section>
      <SectionLabel>{copy.result.why}</SectionLabel>
      <ul className="flex flex-col gap-3">
        {proofs.map((p, i) => (
          <motion.li key={i} initial={{ opacity: 0, x: -6 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.2 + i * 0.06 }} className="grid grid-cols-[96px_1fr] gap-3 text-[15px]">
            <span className="font-mono text-[12px] uppercase tracking-[0.08em] text-source">{p.source}</span>
            <span className="text-text-2">{p.text}</span>
          </motion.li>
        ))}
      </ul>
    </section>
  );
}
