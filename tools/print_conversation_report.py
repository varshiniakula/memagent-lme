# tools/print_conversation_report.py
"""
Pretty-print per-conversation winners from questionwise_*.json.
Usage:
  python tools/print_conversation_report.py --qwise results_lme/questionwise_XXXX.json --sample conv-0012
Omit --sample to print last N conversations.
"""
import json, argparse, re

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwise", required=True)
    ap.add_argument("--sample", default=None)
    ap.add_argument("--last", type=int, default=3)
    args = ap.parse_args()

    data = json.load(open(args.qwise, "r", encoding="utf-8"))
    rows = data["questions"]

    # group by sample
    convs = {}
    for r in rows:
        convs.setdefault(r["sample_id"], []).append(r)

    keys = sorted(convs.keys())
    if args.sample:
        keys = [k for k in keys if k == args.sample]
        if not keys:
            print("Sample not found:", args.sample)
            return
    else:
        keys = keys[-args.last:]

    for sid in keys:
        print(f"\n=== Conversation {sid} ===")
        qs = sorted(convs[sid], key=lambda x: x["question"])
        for q in qs:
            ok = q.get("success_by_retriever", {})
            winners = ",".join(q.get("winner_list") or []) or "-"
            gold = "|".join(q.get("gold_evidence") or [])
            # compact per-retriever flags
            flags = " ".join([f"{name}={'✅' if ok.get(name, False) else '❌'}"
                              for name in sorted(ok.keys())])
            print(f"- Q: {q['question']}")
            print(f"    gold: {gold}")
            print(f"    winners: [{winners}]  oracle={q.get('oracle_success')}")
            print(f"    {flags}")

if __name__ == "__main__":
    main()
