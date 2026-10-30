import { useEffect, useRef, useState } from "react";
import { fmt, onActive, pause, play } from "../audio/player";
import { copy } from "../copy";

export function PreviewPlayer({ src, autoPlay = false, compact = false, label }: { src: string | null; autoPlay?: boolean; compact?: boolean; label?: string }) {
  const ref = useRef<HTMLAudioElement>(null);
  const [playing, setPlaying] = useState(false);
  const [pos, setPos] = useState(0);
  const [dur, setDur] = useState(30);
  useEffect(() => onActive((el) => setPlaying(el === ref.current && !!el)), []);
  useEffect(() => { if (autoPlay && ref.current && src) play(ref.current); }, [autoPlay, src]);
  if (!src) return <div className="text-[14px] text-dim">{copy.result.outOfCatalog}</div>;
  const toggle = () => { const el = ref.current; if (!el) return; if (el.paused) play(el); else pause(el); };
  const size = compact ? "h-10 w-10" : "h-14 w-14";
  return (
    <div className="flex items-center gap-4">
      <audio ref={ref} src={src} preload="metadata" onTimeUpdate={(e) => setPos(e.currentTarget.currentTime)}
        onLoadedMetadata={(e) => setDur(e.currentTarget.duration || 30)} onEnded={() => setPlaying(false)} onPause={() => setPlaying(false)} onPlay={() => setPlaying(true)} />
      <button type="button" onClick={toggle} aria-label={playing ? "Pause" : `Play${label ? ` ${label}` : ""}`}
        className={`flex ${size} shrink-0 items-center justify-center rounded-full bg-accent text-on-accent`}>
        {playing ? <span className="block h-4 w-4 rounded-[3px] bg-on-accent" /> : <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor" aria-hidden><path d="M7 4v16l13-8z" /></svg>}
      </button>
      <div className="flex-1">
        <div className="h-2 w-full overflow-hidden rounded-full bg-track" role="progressbar" aria-valuemin={0} aria-valuemax={dur} aria-valuenow={pos}>
          <div className="h-full bg-accent" style={{ width: `${Math.min(100, (pos / dur) * 100)}%` }} />
        </div>
        {!compact && <div className="mt-2 font-mono text-[12px] text-dim">{copy.result.preview(fmt(pos), fmt(dur))}</div>}
      </div>
    </div>
  );
}
