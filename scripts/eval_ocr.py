import os
import sys

def levenshtein(a, b):
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[m]

def cer(ref, hyp):
    ref_chars, hyp_chars = list(ref), list(hyp)
    return levenshtein(ref_chars, hyp_chars) / max(1, len(ref_chars))

def wer(ref, hyp):
    ref_words, hyp_words = ref.split(), hyp.split()
    return levenshtein(ref_words, hyp_words) / max(1, len(ref_words))

def main(folder):
    pairs = []
    for fn in os.listdir(folder):
        if fn.endswith(".txt"):
            base = fn[:-4]
            gt = os.path.join(folder, base + ".txt")
            out = os.path.join(folder, base + ".out")
            if os.path.exists(out):
                pairs.append((base, gt, out))

    if not pairs:
        print("No pairs found. Expect *.txt and matching *.out files.")
        return 2

    rows = []
    for base, gt_path, out_path in pairs:
        ref = open(gt_path, "r", encoding="utf-8").read().strip()
        hyp = open(out_path, "r", encoding="utf-8").read().strip()
        rows.append((base, cer(ref, hyp), wer(ref, hyp)))

    print("file\tCER\tWER")
    for base, c, w in rows:
        print(f"{base}\t{c:.4f}\t{w:.4f}")

    avg_c = sum(r[1] for r in rows) / len(rows)
    avg_w = sum(r[2] for r in rows) / len(rows)
    print(f"\nAVG\t{avg_c:.4f}\t{avg_w:.4f}")
    return 0

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/eval_ocr.py <folder>")
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
