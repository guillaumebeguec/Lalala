import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { extensionFor, record } from "../audio/recorder";
import { BriefBar, type Chip } from "../components/BriefBar";
import { Header } from "../components/Header";
import { ListenButton, type ListenState } from "../components/ListenButton";
import { ReplayMenu } from "../components/ReplayMenu";
import { copy } from "../copy";
import { useSearch } from "../state/useSearch";

const MAX_FILE = 25 * 1024 * 1024;

export function Capture() {
  const { state, dispatch, start, replay } = useSearch();
  const nav = useNavigate();
  const [text, setText] = useState(state.input.text ?? "");
  const [listen, setListen] = useState<ListenState>("idle");
  const [level, setLevel] = useState(0);
  const [seconds, setSeconds] = useState(0);
  const [notice, setNotice] = useState<string | null>(null);
  const [menu, setMenu] = useState(false);
  const recAbort = useRef<AbortController | null>(null);

  useEffect(() => { setText(state.input.text ?? ""); }, [state.input.text]);

  const chips: Chip[] = [];
  if (state.input.audio) chips.push({ id: "audio", label: state.input.fileName && !state.input.audioDurationS ? state.input.fileName : copy.capture.recordingChip(state.input.audioDurationS ?? 10) });
  if (state.input.video) chips.push({ id: "video", label: state.input.fileName ?? "clip.mp4" });

  const onListen = async () => {
    if (listen === "recording") { recAbort.current?.abort(); return; }
    setListen("recording"); setSeconds(0); setNotice(null);
    const ac = new AbortController(); recAbort.current = ac;
    try {
      const r = await record({ maxS: 10, onLevel: setLevel, onTick: setSeconds, signal: ac.signal });
      dispatch({ type: "setInput", input: { audio: r.blob, audioDurationS: r.durationS, fileName: `recording.${extensionFor(r.mime)}`, video: undefined } });
    } catch { setNotice(copy.capture.micDenied); }
    setListen("idle"); setLevel(0);
  };
  const onFile = (f: File, kind: "video" | "audio") => {
    if (f.size > MAX_FILE) { setNotice(copy.capture.tooBig); return; }
    if (kind === "video") {
      const url = URL.createObjectURL(f); const v = document.createElement("video");
      v.preload = "metadata";
      v.onloadedmetadata = () => { URL.revokeObjectURL(url); if (v.duration > 60) setNotice(copy.capture.tooLong); else dispatch({ type: "setInput", input: { video: f, audio: undefined, fileName: f.name } }); };
      v.onerror = () => dispatch({ type: "setInput", input: { video: f, audio: undefined, fileName: f.name } });
      v.src = url;
    } else dispatch({ type: "setInput", input: { audio: f, video: undefined, fileName: f.name, audioDurationS: undefined } });
  };
  const submit = async () => {
    const input = { ...state.input, text: text.trim() || undefined };
    if (!input.text && !input.audio && !input.video) return;
    const id = Date.now().toString(36);
    nav(`/search/${id}`);
    await start(input);
  };
  return (
    <div className="mx-auto flex min-h-screen max-w-content flex-col px-6 py-6 md:px-10">
      <Header onLogoLongPress={() => setMenu(true)} />
      <main className="flex flex-1 flex-col items-center justify-center gap-10 py-10 text-center">
        <h1 className="font-display text-[44px] font-extrabold leading-[1.02] tracking-[-0.04em] md:text-[64px]">{copy.capture.h1}</h1>
        <ListenButton state={listen} level={level} seconds={seconds} onClick={onListen} />
        {notice && <p role="status" className="text-[15px] text-muted">{notice}</p>}
        <BriefBar value={text} onChange={setText} onSubmit={submit} chips={chips} onRemoveChip={(id) => dispatch({ type: "setInput", input: id === "audio" ? { audio: undefined, audioDurationS: undefined, fileName: undefined } : { video: undefined, fileName: undefined } })}
          onVideo={(f) => onFile(f, "video")} onAudio={(f) => onFile(f, "audio")} disabled={listen === "recording"} />
      </main>
      <ReplayMenu open={menu} onClose={() => setMenu(false)} onPick={async (s) => { setMenu(false); nav(`/search/${s}`); await replay(s); }} />
    </div>
  );
}
