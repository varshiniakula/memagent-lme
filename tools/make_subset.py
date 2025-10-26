# tools/make_subset.py
import json, argparse, random, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="path to longmemeval_*_cleaned.json")
    ap.add_argument("--out", required=True, help="output json (subset)")
    ap.add_argument("--n", type=int, default=20, help="how many items to keep")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.src, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected a JSON list at top-level.")

    random.seed(args.seed)
    if args.n < len(data):
        idx = list(range(len(data)))
        random.shuffle(idx)
        pick = sorted(idx[:args.n])
        data = [data[i] for i in pick]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[OK] wrote {len(data)} items -> {args.out}")

if __name__ == "__main__":
    main()
