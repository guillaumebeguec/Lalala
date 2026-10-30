import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import basicSsl from "@vitejs/plugin-basic-ssl";

// LAN=1 : exposé sur toutes les interfaces en HTTPS auto-signé (le micro exige un contexte sécurisé).
const lan = process.env.LAN === "1";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), ...(lan ? [basicSsl()] : [])],
  server: { host: lan ? true : undefined, https: lan ? {} : undefined, proxy: { "/api": "http://localhost:8000" } },
  build: { outDir: "dist" },
  test: {
    environment: "jsdom",
    setupFiles: ["./tests/setup.ts"],
    include: ["tests/**/*.test.{ts,tsx}"],
    globals: true,
  },
});
