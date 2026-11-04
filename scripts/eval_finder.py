"""Evaluation of the agent with a simulated user (PLAN.md phases 6.4 and 7) → eval/finder.md, eval/finder_ig.md.

    uv run python scripts/eval_finder.py --subset dev --sets t1,t2 --policies baseline,ig --users plain,informed [--limit N]

For each case: the agent receives the description, asks its questions to the simulated user
(who only sees the description), then proposes. Measures: top-1, top-3 (the right answer among the
proposed candidates, in order), number of questions, tool calls, time, share of answers
found via the web. Bootstrap confidence intervals on top-1.
The anti-leak rule is structural: neither the agent nor the simulated user ever receives `reponse_id`.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catalog import build as cb  # noqa: E402
from core.config import PROJECT_ROOT  # noqa: E402
from core.metrics import bootstrap_ci, summarize  # noqa: E402
from finder.agent import Agent  # noqa: E402
from finder.belief import ig_policy  # noqa: E402
from finder.simulated_user import SimulatedUser  # noqa: E402
from finder.tools import Toolbox  # noqa: E402
from testsets.common import read_jsonl, testsets_dir  # noqa: E402

log = logging.getLogger("eval_finder")


def run_case(tools: Toolbox, case: dict, policy: str, user_kind: str, meta: dict, max_turns: int = 8) -> dict:
    tb_meta = meta.get(case["reponse_id"], {})
    user = SimulatedUser(description=case["texte"], informed=(user_kind == "informed"),
                         known_year=tb_meta.get("year") if user_kind == "informed" else None,
                         known_language=tb_meta.get("language") if user_kind == "informed" else None)
    agent = Agent(tools, question_policy=ig_policy() if policy == "ig" else None)
    t0 = time.time()
    events = list(agent.step(user.opening_message()))
    turns = 0
    entropies = []
    while agent.state.pending_question is not None and turns < max_turns:
        q = agent.state.pending_question
        ans = user.answer(q["question"], [o for o in q["options"] if o != "je ne sais pas"])
        events += list(agent.step(ans))
        turns += 1
    for e in events:
        if e.type == "belief":
            entropies.append(e.data["entropy"])
    b = getattr(agent.state, "belief", None)
    hist = b.history if b else []
    proposals = agent.state.proposals[-1] if agent.state.proposals else []
    ids = [p["track_id"] for p in proposals]
    rank = ids.index(case["reponse_id"]) + 1 if case["reponse_id"] in ids else None
    return {"id": case["id"], "rank": rank, "n_questions": len(agent.state.questions), "tool_calls": agent.state.tool_calls,
            "seconds": round(time.time() - t0, 1), "via_web": any(p.get("source") == "web" for p in proposals),
            "entropy_drops": sum(1 for h in hist if h["entropy_after"] < h["entropy_before"]), "n_updates": len(hist),
            "answers": [q.get("answer") for q in agent.state.questions]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", choices=["dev", "full"], default="dev")
    ap.add_argument("--sets", default="t1,t2")
    ap.add_argument("--policies", default="baseline,ig")
    ap.add_argument("--users", default="plain,informed")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-rerank", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for name in ("httpx", "httpcore", "openai"):
        logging.getLogger(name).setLevel(logging.WARNING)
    tools = Toolbox(subset=args.subset, use_rerank=not args.no_rerank)
    cat = cb.load(args.subset)
    meta = {r.track_id: {"year": int(r.year) if r.year == r.year and r.year is not None else None, "language": r.language} for r in cat.itertuples(index=False)}
    ids = set(cat.track_id)
    d = testsets_dir()
    rows = []
    report = [f"# Agent — evaluation with a simulated user ({args.subset} subset)", ""]
    for key, fname in (("t1", "t1_tot_eval.jsonl"), ("t2", "t2_fr_memories.jsonl")):
        if key not in args.sets.split(",") or not (d / fname).exists():
            continue
        cases = [c for c in read_jsonl(d / fname) if c["reponse_id"] in ids]
        if args.limit:
            cases = cases[: args.limit]
        if not cases:
            continue
        report += [f"## {key.upper()} ({len(cases)} cases)", "", "| Policy | User | top-1 [95 % CI] | top-3 | questions | tools | s/case | entropy ↓ |", "|---|---|---|---|---|---|---|---|"]
        for policy in args.policies.split(","):
            for user_kind in args.users.split(","):
                res = []
                for i, c in enumerate(cases, 1):
                    try:
                        r = run_case(tools, c, policy, user_kind, meta)
                    except Exception as e:  # noqa: BLE001
                        log.warning("%s %s : %s", policy, c["id"], e)
                        r = {"id": c["id"], "rank": None, "n_questions": 0, "tool_calls": 0, "seconds": 0.0, "via_web": False, "entropy_drops": 0, "n_updates": 0, "answers": []}
                    r.update(set=key, policy=policy, user=user_kind)
                    res.append(r)
                    rows.append(r)
                    if i % 10 == 0:
                        m = summarize([x["rank"] for x in res], ks=(1, 3))
                        log.info("%s %s %s : %d / %d, top-1 %.2f, top-3 %.2f", key, policy, user_kind, i, len(cases), m["top1"], m["top3"])
                ranks = [x["rank"] for x in res]
                m = summarize(ranks, ks=(1, 3))
                lo, hi = bootstrap_ci(ranks, 1)
                nq = sum(x["n_questions"] for x in res) / len(res)
                nt = sum(x["tool_calls"] for x in res) / len(res)
                sec = sum(x["seconds"] for x in res) / len(res)
                ups = sum(x["n_updates"] for x in res)
                drops = sum(x["entropy_drops"] for x in res)
                report.append(f"| {policy} | {user_kind} | {m['top1']:.3f} [{lo:.2f}, {hi:.2f}] | {m['top3']:.3f} | {nq:.2f} | {nt:.2f} | {sec:.0f} | {drops}/{ups} |")
        report.append("")
    out = PROJECT_ROOT / "eval" / ("finder_ig.md" if "ig" in args.policies else "finder.md")
    out.write_text("\n".join(report), encoding="utf-8")
    with (PROJECT_ROOT / "eval" / "finder_cases.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["set", "policy", "user", "id", "rank", "n_questions", "tool_calls", "seconds", "via_web", "entropy_drops", "n_updates", "answers"])
        w.writeheader()
        for r in rows:
            r = dict(r); r["answers"] = json.dumps(r["answers"], ensure_ascii=False)
            w.writerow(r)
    print(f"\nReport: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
