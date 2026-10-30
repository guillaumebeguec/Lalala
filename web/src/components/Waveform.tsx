/** Forme d'onde réduite (~64 valeurs 0–1) ; la partie « analysée » en accent. */
export function Waveform({ values, analysed = 1, height = 56 }: { values: number[]; analysed?: number; height?: number }) {
  const n = values.length || 1;
  return (
    <div className="flex items-center gap-[3px]" style={{ height }} aria-hidden>
      {values.map((v, i) => (
        <span key={i} className={`block w-[4px] rounded-full ${i / n <= analysed ? "bg-accent" : "bg-track"}`} style={{ height: `${Math.max(8, v * 100)}%` }} />
      ))}
    </div>
  );
}
