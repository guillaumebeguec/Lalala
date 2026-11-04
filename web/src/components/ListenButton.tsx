import { motion, useReducedMotion } from "framer-motion";
import { copy } from "../copy";

export type ListenState = "idle" | "recording" | "disabled";

/** 168 px disc inside a 300 px area: dotted circle + yellow circle + 3 decorative dots. */
export function ListenButton({ state, level = 0, seconds = 0, onClick }: { state: ListenState; level?: number; seconds?: number; onClick: () => void }) {
  const reduce = useReducedMotion();
  const recording = state === "recording";
  const scale = recording ? 1 + level * 0.12 : 1;
  const progress = Math.min(1, seconds / 10);
  const r = 140, c = 2 * Math.PI * r;
  return (
    <div className="relative mx-auto flex h-[300px] w-[300px] items-center justify-center">
      <svg className="absolute inset-0" viewBox="0 0 300 300" aria-hidden>
        <circle cx="150" cy="150" r={r} fill="none" stroke="#4A3F7A" strokeWidth="2" strokeDasharray="4 10" />
        {recording && (
          <circle cx="150" cy="150" r={r} fill="none" stroke="var(--accent)" strokeWidth="4" strokeLinecap="round"
            strokeDasharray={c} strokeDashoffset={c * (1 - progress)} transform="rotate(-90 150 150)" />
        )}
        <circle cx="52" cy="78" r="4" fill="var(--accent)" />
        <circle cx="250" cy="62" r="4" fill="#FFE066" />
        <circle cx="232" cy="236" r="4" fill="#F7F3FF" />
      </svg>
      <motion.button
        type="button"
        aria-label={recording ? copy.capture.stop : copy.capture.listen}
        aria-pressed={recording}
        disabled={state === "disabled"}
        onClick={onClick}
        className="relative flex h-[168px] w-[168px] items-center justify-center rounded-full bg-yellow text-on-accent shadow-[0_18px_50px_rgba(255,224,102,0.25)] disabled:opacity-50"
        animate={reduce ? {} : recording ? { scale } : { scale: [1, 1.03, 1] }}
        transition={recording ? { duration: 0.08 } : { duration: 3, repeat: Infinity, ease: "easeInOut" }}
      >
        {recording ? (
          <span className="block h-12 w-12 rounded-[10px] bg-on-accent" />
        ) : (
          <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
            <rect x="9" y="2" width="6" height="12" rx="3" />
            <path d="M5 10a7 7 0 0 0 14 0M12 17v5M8 22h8" />
          </svg>
        )}
      </motion.button>
      {recording && (
        <div className="absolute -bottom-2 font-mono text-[15px] text-text-2" aria-live="polite">
          0:{seconds.toString().padStart(2, "0")}
        </div>
      )}
    </div>
  );
}
