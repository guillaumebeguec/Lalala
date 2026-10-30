import { motion } from "framer-motion";
import type { Confidence, Track } from "../api/types";
import { copy } from "../copy";
import { Cover } from "./Cover";
import { PreviewPlayer } from "./PreviewPlayer";

export function TrackHero({ track, confidence, autoPlay }: { track: Track; confidence: Confidence; autoPlay: boolean }) {
  const meta = [track.artist, track.album, track.year].filter(Boolean).join(" · ");
  return (
    <div className="flex flex-col gap-6 md:flex-row md:items-start">
      <motion.div layoutId={`cover-${track.id}`} initial={{ scale: 0.9, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ duration: 0.3 }}>
        <Cover src={track.cover_url} size={240} className="border-2 border-accent" />
      </motion.div>
      <div className="min-w-0 flex-1">
        <span className="inline-block rounded-pill bg-yellow px-3 py-1 font-mono text-[12px] font-medium text-on-accent">{copy.result.badge[confidence]}</span>
        <motion.h1 initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }} className="mt-3 font-display text-[40px] font-extrabold leading-[1.05] tracking-[-0.04em] md:text-[54px]">{track.title}</motion.h1>
        <div className="mt-2 text-[17px] text-text-2">{meta}</div>
        <div className="mt-5">
          <PreviewPlayer src={track.preview_url} autoPlay={autoPlay} label={track.title} />
          {!track.preview_url && track.search_url && <a className="text-[14px] text-source underline" href={track.search_url} target="_blank" rel="noreferrer">{copy.result.outOfCatalog}</a>}
        </div>
      </div>
    </div>
  );
}
