import { motion } from "framer-motion";
import { useRef } from "react";
import type { Track } from "../api/types";
import { play } from "../audio/player";
import { Cover } from "./Cover";

export function CandidateRow({ track, onPromote }: { track: Track; onPromote: (id: string) => void }) {
  const audio = useRef<HTMLAudioElement>(null);
  return (
    <motion.li layout className="flex items-center gap-3 rounded-row border border-border bg-surface p-3">
      <button type="button" onClick={() => onPromote(track.id)} className="flex min-h-[44px] flex-1 items-center gap-3 text-left">
        <motion.div layoutId={`cover-${track.id}`}><Cover src={track.cover_url} size={56} rounded="rounded-thumb" /></motion.div>
        <span className="min-w-0">
          <span className="block truncate text-[16px] font-medium text-text">{track.title}</span>
          <span className="block truncate text-[14px] text-text-2">{[track.artist, track.year].filter(Boolean).join(" · ")}</span>
        </span>
      </button>
      {track.preview_url && (
        <>
          <audio ref={audio} src={track.preview_url} preload="none" />
          <button type="button" aria-label={`Play ${track.title}`} onClick={() => audio.current && (audio.current.paused ? play(audio.current) : audio.current.pause())}
            className="flex h-11 w-11 items-center justify-center rounded-full border border-border-strong text-text-2">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden><path d="M7 4v16l13-8z" /></svg>
          </button>
        </>
      )}
    </motion.li>
  );
}
