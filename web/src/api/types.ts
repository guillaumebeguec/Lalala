// Contrat d'API (FRONT_PLAN.md § 4).
export type Source = "catalog" | "web" | "fingerprint";
export type Confidence = "high" | "medium" | "low";
export type InputKind = "audio" | "video" | "text" | "mixed";
export type StepStatus = "pending" | "running" | "done" | "skipped";

export interface Track {
  id: string;
  title: string;
  artist: string;
  album?: string | null;
  year?: number | null;
  cover_url: string | null;
  preview_url: string | null;
  source: Source;
  search_url?: string | null;
  in_catalog?: boolean;
}

export interface ScoreRow {
  track_id: string;
  bm25?: number;
  emb: Record<string, { score: number; rank: number }>;
  fingerprint?: number;
  rerank?: number;
  final: number;
  source?: string;
}

export interface InputEvent {
  kind: InputKind;
  duration_s?: number;
  noise_detected?: boolean;
  note?: string | null;
  waveform: number[];
}

export interface StepEvent {
  id: string;
  title: string;
  model: string;
  status: StepStatus;
  detail: string;
  elapsed_ms?: number;
}

export interface ResultEvent {
  query_id: string;
  confidence: Confidence;
  best: Track | null;
  alternatives: Track[];
  proofs: { source: string; text: string }[];
  scores: ScoreRow[];
  timings: { step: string; ms: number }[];
  analysis: Record<string, unknown>;
}

export type SearchEvent =
  | { type: "input"; data: InputEvent }
  | { type: "step"; data: StepEvent }
  | { type: "result"; data: ResultEvent }
  | { type: "error"; data: { message: string } };

export interface SearchInput {
  text?: string;
  audio?: Blob;
  video?: Blob;
  audioDurationS?: number;
  fileName?: string;
}

export interface SearchApi {
  /** Lance une recherche ; `onEvent` reçoit le flux ; la promesse se résout à la fin du flux. */
  search(input: SearchInput, onEvent: (e: SearchEvent) => void, signal?: AbortSignal): Promise<void>;
  replay(scene: string, onEvent: (e: SearchEvent) => void, signal?: AbortSignal): Promise<void>;
  feedback(payload: { query_id: string; track_id: string | null; verdict: "match" | "reject" }): Promise<void>;
}
