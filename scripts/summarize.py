"""Collect DrQ+BC results into a markdown table (final5 mean +- std over seeds), next to the paper's numbers.

    python scripts/summarize.py [--runs runs] [--csv results.csv]
"""
import argparse
import csv
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path

DOMAINS = ['cheetah', 'walker']
SETTINGS = ['clean', 'static', 'easy_video', 'hard_video']
# DrQ+BC final5, mean +- std over 4 seeds, 5 pretraining policies (paper table).
PAPER = {
    'acroaf': {'cheetah': [(523, 10), (614, 8), (449, 32), (293, 21)],
               'walker': [(885, 34), (649, 67), (578, 17), (443, 10)]},
    'polcon': {'cheetah': [(402, 80), (425, 30), (336, 51), (196, 26)],
               'walker': [(726, 42), (384, 87), (378, 19), (354, 36)]},
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', type=Path, default=Path(os.environ.get('CARDPOL_RUNS', 'runs')))
    p.add_argument('--csv', type=Path, default=None, help='also write per-seed rows here')
    args = p.parse_args()

    scores, rows = defaultdict(list), []
    for path in sorted(args.runs.glob('pretrain/*/*/**/drqbc/seed*/result.json')):
        r = json.loads(path.read_text())
        scores[(r['method'], r['domain'], r['setting'])].append(r['final5'])
        rows.append({**r, 'path': str(path.parent)})
    if not rows:
        print(f'no results under {args.runs}/pretrain/*/*/.../drqbc/seed*/result.json yet')
        return
    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    cols = [(d, s) for d in DOMAINS for s in SETTINGS]
    print('| Method | ' + ' | '.join(f'{d} {s}' for d, s in cols) + ' |')
    print('|---' * (len(cols) + 1) + '|')
    for method in sorted({k[0] for k in scores}):
        cells = []
        for d, s in cols:
            v = scores.get((method, d, s), [])
            std = statistics.stdev(v) if len(v) > 1 else 0.0
            cells.append(f'{statistics.mean(v):.0f} ± {std:.0f} ({len(v)})' if v else '–')
        print(f'| {method} | ' + ' | '.join(cells) + ' |')
        if method in PAPER:
            paper = [PAPER[method][d][SETTINGS.index(s)] for d, s in cols]
            print(f'| {method} (paper) | ' + ' | '.join(f'{m} ± {sd} (4)' for m, sd in paper) + ' |')
    print('\ncells: DrQ+BC final5 (mean of the last 5 evaluations), mean ± std over seeds (n)')


if __name__ == '__main__':
    main()
