import { motion, useReducedMotion } from "framer-motion";

/** Three dots (pink, yellow, white): bouncing in sequence, 900 ms per cycle. */
export function ThreeDots({ size = 14 }: { size?: number }) {
  const reduce = useReducedMotion();
  const colors = ["bg-accent", "bg-yellow", "bg-text"];
  return (
    <div className="flex items-end gap-2" role="status" aria-label="Loading">
      {colors.map((c, i) => (
        <motion.span
          key={c}
          className={`inline-block rounded-full ${c}`}
          style={{ width: size, height: size }}
          animate={reduce ? { opacity: [0.4, 1, 0.4] } : { y: [0, -10, 0] }}
          transition={{ duration: 0.9, repeat: Infinity, delay: i * 0.15, ease: "easeInOut" }}
        />
      ))}
    </div>
  );
}
