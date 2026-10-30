import { copy } from "../copy";
import { PreviewPlayer } from "./PreviewPlayer";

export function CompareCards({ yours, original }: { yours: string | null; original: string | null }) {
  return (
    <div className="grid gap-4 md:grid-cols-2">
      <div className="rounded-card border border-border bg-surface p-5">
        <div className="mb-3 font-mono text-[12px] uppercase tracking-[0.1em] text-muted">{copy.result.yourRecording}</div>
        <PreviewPlayer src={yours} compact label={copy.result.yourRecording} />
      </div>
      <div className="rounded-card bg-surface-strong p-5">
        <div className="mb-3 font-mono text-[12px] uppercase tracking-[0.1em] text-accent-soft">{copy.result.original}</div>
        <PreviewPlayer src={original} compact label={copy.result.original} />
      </div>
    </div>
  );
}
