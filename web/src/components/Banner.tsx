import { copy } from "../copy";

export function Banner({ onRetry }: { onRetry: () => void }) {
  return (
    <div role="alert" className="mb-6 flex items-center justify-between gap-4 rounded-row border border-accent bg-surface px-5 py-3">
      <span className="text-[15px] text-text">{copy.error.banner}</span>
      <button type="button" onClick={onRetry} className="rounded-pill bg-yellow px-4 py-2 text-[14px] font-medium text-on-accent">{copy.error.retry}</button>
    </div>
  );
}
