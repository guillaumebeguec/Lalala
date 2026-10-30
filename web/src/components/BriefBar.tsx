import { useRef } from "react";
import { copy } from "../copy";

export interface Chip { id: string; label: string }

export function BriefBar({ value, onChange, onSubmit, chips, onRemoveChip, onVideo, onAudio, disabled }: {
  value: string; onChange: (v: string) => void; onSubmit: () => void; chips: Chip[]; onRemoveChip: (id: string) => void;
  onVideo: (f: File) => void; onAudio: (f: File) => void; disabled?: boolean;
}) {
  const videoIn = useRef<HTMLInputElement>(null);
  const audioIn = useRef<HTMLInputElement>(null);
  return (
    <div className="mx-auto w-full max-w-[760px]">
      {chips.length > 0 && (
        <div className="mb-3 flex flex-wrap gap-2">
          {chips.map((c) => (
            <span key={c.id} className="inline-flex items-center gap-2 rounded-pill bg-chip px-4 py-2 font-mono text-[13px] text-text-2">
              {c.label}
              <button type="button" aria-label={`${copy.capture.remove} ${c.label}`} onClick={() => onRemoveChip(c.id)} className="text-muted hover:text-text">✕</button>
            </span>
          ))}
        </div>
      )}
      <form className="flex items-center gap-3" onSubmit={(e) => { e.preventDefault(); onSubmit(); }}>
        <div className="flex flex-1 items-center rounded-pill border border-border bg-surface py-2 pl-6 pr-2">
          <input
            aria-label={copy.capture.describe}
            placeholder={copy.capture.placeholder}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            disabled={disabled}
            className="min-h-[44px] flex-1 bg-transparent text-[17px] text-text placeholder:text-dim focus:outline-none"
          />
          <button type="submit" aria-label={copy.capture.search} disabled={disabled} className="flex h-11 w-11 items-center justify-center rounded-full bg-yellow text-on-accent disabled:opacity-50">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden><path d="M5 12h14M13 5l7 7-7 7" /></svg>
          </button>
        </div>
        <IconButton label={copy.capture.addVideo} onClick={() => videoIn.current?.click()}>
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden><rect x="3" y="6" width="13" height="12" rx="2" /><path d="m16 10 5-3v10l-5-3z" /></svg>
        </IconButton>
        <IconButton label={copy.capture.addAudio} onClick={() => audioIn.current?.click()}>
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden><path d="M9 18V6l11-2v12" /><circle cx="6" cy="18" r="3" /><circle cx="17" cy="16" r="3" /></svg>
        </IconButton>
        <input ref={videoIn} type="file" accept="video/*" className="hidden" onChange={(e) => { const f = e.target.files?.[0]; if (f) onVideo(f); e.target.value = ""; }} />
        <input ref={audioIn} type="file" accept="audio/*" className="hidden" onChange={(e) => { const f = e.target.files?.[0]; if (f) onAudio(f); e.target.value = ""; }} />
      </form>
    </div>
  );
}

function IconButton({ label, onClick, children }: { label: string; onClick: () => void; children: React.ReactNode }) {
  return (
    <button type="button" aria-label={label} onClick={onClick} className="flex h-12 w-12 items-center justify-center rounded-full border border-border-strong bg-surface text-text-2 hover:text-text">
      {children}
    </button>
  );
}
