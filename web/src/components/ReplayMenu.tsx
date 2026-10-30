import { copy } from "../copy";

export function ReplayMenu({ open, onClose, onPick }: { open: boolean; onClose: () => void; onPick: (scene: string) => void }) {
  if (!open) return null;
  const scenes = Object.entries(copy.replay.scenes) as [string, string][];
  return (
    <div className="fixed inset-0 z-30 flex items-start justify-start bg-bg/60 p-6" onClick={onClose}>
      <div role="menu" aria-label={copy.replay.title} className="rounded-card border border-border bg-surface p-3" onClick={(e) => e.stopPropagation()}>
        <div className="px-3 py-1 font-mono text-[12px] uppercase tracking-[0.1em] text-muted">{copy.replay.title}</div>
        {scenes.map(([k, label]) => (
          <button key={k} role="menuitem" type="button" onClick={() => onPick(k)} className="block w-full rounded-row px-3 py-2 text-left text-[15px] hover:bg-chip">{label}</button>
        ))}
      </div>
    </div>
  );
}
