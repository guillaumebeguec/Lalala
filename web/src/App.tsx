import { useEffect } from "react";
import { Route, Routes, useNavigate, useSearchParams } from "react-router-dom";
import { Capture } from "./screens/Capture";
import { Result } from "./screens/Result";
import { Searching } from "./screens/Searching";
import { SearchProvider, useSearch } from "./state/useSearch";

/** `?replay=memory|cover|noisy` and `?scale=1.25` (FRONT_PLAN.md § 7). */
function QueryOptions() {
  const [params] = useSearchParams();
  const { replay } = useSearch();
  const nav = useNavigate();
  useEffect(() => {
    const scale = params.get("scale");
    if (scale) document.documentElement.style.setProperty("--scale", scale);
    const accent = params.get("accent");
    if (accent) document.documentElement.dataset.accent = accent;
    const scene = params.get("replay");
    if (scene) { nav(`/search/${scene}`, { replace: true }); void replay(scene); }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return null;
}

export default function App() {
  return (
    <SearchProvider>
      <QueryOptions />
      <Routes>
        <Route path="/" element={<Capture />} />
        <Route path="/search/:id" element={<Searching />} />
        <Route path="/result/:id" element={<Result />} />
        <Route path="*" element={<Capture />} />
      </Routes>
    </SearchProvider>
  );
}
