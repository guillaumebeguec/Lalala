import { copy } from "../copy";

/** Three staggered "la" (pink, yellow, white), shifted up by 0 / 7 / 14 px. */
export function Logo({ onLongPress }: { onLongPress?: () => void }) {
  let timer: number | undefined;
  const start = () => { if (onLongPress) timer = window.setTimeout(onLongPress, 1000); };
  const stop = () => { if (timer) window.clearTimeout(timer); };
  return (
    <div
      className="inline-flex select-none items-end font-display text-[28px] font-extrabold tracking-[-0.03em]"
      aria-label={copy.logo}
      onPointerDown={start}
      onPointerUp={stop}
      onPointerLeave={stop}
    >
      <span className="text-accent">la</span>
      <span className="relative -top-[7px] text-yellow">la</span>
      <span className="relative -top-[14px] text-text">la</span>
    </div>
  );
}
