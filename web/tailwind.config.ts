import type { Config } from "tailwindcss";

// Tokens de FRONT_PLAN.md § 2 : aucune couleur en dur dans les composants.
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#15122B",
        surface: "#211C3D",
        "surface-strong": "#2E2347",
        chip: "#2A2450",
        border: "#342D5C",
        "border-strong": "#4A3F7A",
        track: "#3C3466",
        text: "#F7F3FF",
        "text-2": "#DCD6F2",
        muted: "#B9B2D6",
        dim: "#9C94BF",
        source: "#B4A9FF",
        accent: "var(--accent)",
        "accent-soft": "var(--accent-soft)",
        yellow: "#FFE066",
        "on-accent": "#15122B",
      },
      fontFamily: {
        display: ['"Bricolage Grotesque"', "system-ui", "sans-serif"],
        sans: ['"DM Sans"', "system-ui", "sans-serif"],
        mono: ['"DM Mono"', "ui-monospace", "monospace"],
      },
      borderRadius: { pill: "999px", card: "24px", hero: "28px", row: "18px", thumb: "14px" },
      maxWidth: { content: "1160px" },
    },
  },
  plugins: [],
} satisfies Config;
