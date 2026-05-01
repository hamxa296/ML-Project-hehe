#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fraud_ensemble_paper.src.paper_pipeline import train_paper_style

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/merged_raw_train.csv')
    p.add_argument('--sample', type=int, default=200000)
    args = p.parse_args()
    path = Path(args.data)
    res = train_paper_style(str(path), sample=args.sample)
    print('Done. Meta model:', res['meta'])

if __name__ == '__main__':
    main()
