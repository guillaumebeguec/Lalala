import { useState } from "react";

export function Cover({ src, size, className = "", rounded = "rounded-hero" }: { src: string | null; size: number; className?: string; rounded?: string }) {
  const [err, setErr] = useState(false);
  if (!src || err) return <div className={`${rounded} bg-chip ${className}`} style={{ width: size, height: size }} aria-hidden />;
  return <img src={src} alt="" width={size} height={size} onError={() => setErr(true)} className={`${rounded} object-cover ${className}`} style={{ width: size, height: size }} />;
}
