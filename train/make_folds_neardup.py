"""ЧЕСТНЫЕ фолды: группировка почти-дубликатов, а не только точных копий.

  python train/make_folds_neardup.py [--tau 0.95]

Зачем. Старые фолды группировали только точные копии (хэш name+desc).
Диагностика diag_neardup.py показала: 70% строк имеют почти-дубль
(косинус e5 >= 0.95) в ЧУЖИХ фолдах, и на них метрика 0.945, а на чистых
строках — 0.828. Домашние 0.91 были памятью, а не умением; сервер
(«в тестовых данных будут другие товары») дал 0.719.

Здесь строки объединяются в компоненты связности по рёбрам:
  * точное совпадение нормализованного name+desc (как раньше);
  * косинус e5-эмбеддингов >= tau внутри категории.
Компонента целиком едет в один фолд. OOF после этого измеряет умение
обобщать на действительно новые товары.
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CAT_BAD, CAT_FLAM  # noqa: E402
from src.data_prep import prepare_dataframe  # noqa: E402
from src.dupes import exact_key  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "data" / "work"
N_FOLDS = 5
SEED = 42


class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau", type=float, default=0.95)
    args = parser.parse_args()

    df = prepare_dataframe(ROOT / "data" / "data.csv")
    emb = np.load(WORK / "e5_embeddings.npy").astype(np.float32)
    assert len(emb) == len(df)
    cats = df["category"].values
    labels = df["label"].values
    dsu = DSU(len(df))

    # рёбра 1: точные копии
    seen = {}
    for i, (n, d) in enumerate(zip(df["name"].fillna(""),
                                   df["description"].fillna(""))):
        k = exact_key(str(n), str(d))
        if k in seen:
            dsu.union(seen[k], i)
        else:
            seen[k] = i

    # рёбра 2: почти-дубликаты по e5 внутри категории
    for cat in (CAT_BAD, CAT_FLAM):
        idx = np.where(cats == cat)[0]
        be = emb[idx]
        for s in range(0, len(idx), 512):
            sims = be[s:s + 512] @ be.T
            rows, cols = np.where(sims >= args.tau)
            for r, c in zip(rows, cols):
                gi, gj = idx[s + r], idx[c]
                if gi != gj:
                    dsu.union(gi, gj)

    comp = defaultdict(list)
    for i in range(len(df)):
        comp[dsu.find(i)].append(i)
    comps = sorted(comp.values(), key=len, reverse=True)
    sizes = [len(c) for c in comps]
    print(f"[folds2] tau={args.tau}: компонент {len(comps)}, "
          f"топ-5 размеров {sizes[:5]}, строк в топ-1 "
          f"{sizes[0]/len(df)*100:.1f}%")

    # раскладка: компонента целиком в один фолд; жадно балансируем
    # размер и число label=1 в разрезе категорий
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(comps))
    order = sorted(order, key=lambda i: -sizes[i])
    fold_size = np.zeros((N_FOLDS, 2))          # [fold, cat]
    fold_pos = np.zeros((N_FOLDS, 2))
    cat_idx = {CAT_BAD: 0, CAT_FLAM: 1}
    fold_of = np.full(len(df), -1, dtype=int)
    for ci in order:
        rows = comps[ci]
        add_size = np.zeros(2)
        add_pos = np.zeros(2)
        for i in rows:
            c = cat_idx[cats[i]]
            add_size[c] += 1
            add_pos[c] += labels[i]
        # штраф: перекос размера + перекос доли позитивов
        best_f, best_cost = 0, None
        for f in range(N_FOLDS):
            ns = fold_size[f] + add_size
            npos = fold_pos[f] + add_pos
            cost = ns.sum()
            for c in range(2):
                total_pos = fold_pos[:, c].sum() + add_pos[c]
                total = fold_size[:, c].sum() + add_size[c]
                if ns[c] > 0 and total > 0:
                    cost += 3.0 * abs(npos[c] / max(ns[c], 1) -
                                      total_pos / total) * ns[c]
            if best_cost is None or cost < best_cost:
                best_f, best_cost = f, cost
        for i in rows:
            fold_of[i] = best_f
        c = cat_idx[cats[rows[0]]]
        fold_size[best_f] += add_size
        fold_pos[best_f] += add_pos

    assert (fold_of >= 0).all()
    out = pd.DataFrame({"id": df["id"], "fold": fold_of})
    old = WORK / "folds.csv"
    if old.exists():
        backup = WORK / "folds_exact_v1.csv"
        if not backup.exists():
            old.rename(backup)
            print(f"[folds2] старые фолды -> {backup.name}")
    out.to_csv(WORK / "folds.csv", index=False)

    for f in range(N_FOLDS):
        m = fold_of == f
        parts = []
        for cat in (CAT_BAD, CAT_FLAM):
            cm = m & (cats == cat)
            parts.append(f"{cat[:3]}: {cm.sum()} (ok {labels[cm].mean()*100:.1f}%)")
        print(f"[folds2] фолд {f}: {m.sum()} строк | " + " | ".join(parts))
    print(f"[folds2] записан {WORK / 'folds.csv'}")


if __name__ == "__main__":
    main()
