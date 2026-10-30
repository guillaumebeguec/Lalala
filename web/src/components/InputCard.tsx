import type { InputEvent } from "../api/types";
import { copy } from "../copy";
import { Waveform } from "./Waveform";
import { SectionLabel } from "./SectionLabel";

export function InputCard({ info, analysed }: { info: InputEvent | null; analysed: number }) {
  const kind = info?.kind ?? "text";
  const line = copy.searching.kind[kind](info?.duration_s);
  return (
    <section className="rounded-card border border-border bg-surface p-6">
      <SectionLabel>{copy.searching.yourInput}</SectionLabel>
      {info?.noise_detected && <span className="mb-3 inline-block rounded-pill bg-yellow px-3 py-1 font-mono text-[12px] font-medium text-on-accent">{copy.searching.noise}</span>}
      {info && info.waveform.length > 0 && <Waveform values={info.waveform} analysed={analysed} />}
      <div className="mt-3 text-[15px] text-text-2">{line}</div>
      {info?.note && <div className="mt-1 text-[15px] text-muted">{copy.searching.note(info.note)}</div>}
    </section>
  );
}
