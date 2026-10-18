"""Диагностика: насколько OOF раздут почти-дубликатами через границы фолдов.

Для каждой строки считаем максимальный косинус e5 к строкам ЧУЖИХ фолдов
своей категории. Метрика отдельно на строках с кросс-фолдовым почти-дублем
и на «чистых». Если на чистых метрика ~= серверному скору — вот и разгадка
разрыва 0.91 дома против 0.72 на сервере.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import ARTIFACTS_DIR, CAT_BAD, CAT_FLAM  # noqa: E402
from train.cv_utils import WORK_DIR, load_or_make_folds  # noqa: E402
from train.metric import metric_a  # noqa: E402
from train.tune_blend import blend_arrays, load_oof  # noqa: E402

TAUS = [0.90, 0.95, 0.97, 0.99]


def main():
    df = load_or_make_folds(use_ocr=False)
    y = df["label"].values
    cats = df["category"].values
    folds = df["fold"].values
    emb = np.load(WORK_DIR / "e5_embeddings.npy").astype(np.float32)

    # максимальный косинус к чужим фолдам своей категории
    maxcos = np.zeros(len(df), dtype=np.float32)
    for cat in (CAT_BAD, CAT_FLAM):
        cm = cats == cat
        for f in sorted(set(folds)):
            q = cm & (folds == f)
            base = cm & (folds != f)
            if not q.any() or not base.any():
                continue
            be = emb[base]
            qi = np.where(q)[0]
            for s in range(0, len(qi), 512):
                chunk = qi[s:s + 512]
                sims = emb[chunk] @ be.T
                maxcos[chunk] = sims.max(axis=1)

    # OOF-вероятности полного профиля (та же формула, что в проде)
    cfg = json.loads((ARTIFACTS_DIR / "blend.json").read_text(encoding="utf-8"))
    prof = cfg["profiles"][0]
    sources = load_oof(df)
    probs = np.zeros(len(df))
    for cat in (CAT_BAD, CAT_FLAM):
        m = cats == cat
        cat_sources = {k: v[m] for k, v in sources.items()}
        probs[m] = blend_arrays(cat_sources, prof["weights"][cat], int(m.sum()))
    preds = np.array([int(probs[i] >= prof["thresholds"][cats[i]])
                      for i in range(len(df))])

    print(f"OOF A на всех строках: {metric_a(y, preds, cats):.4f}")
    for tau in TAUS:
        dup_m = maxcos >= tau
        clean_m = ~dup_m
        a_dup = metric_a(y[dup_m], preds[dup_m], cats[dup_m]) if dup_m.any() else float("nan")
        a_clean = metric_a(y[clean_m], preds[clean_m], cats[clean_m]) if clean_m.any() else float("nan")
        print(f"tau={tau:.2f}: почти-дублей {dup_m.mean()*100:5.1f}% "
              f"(A={a_dup:.4f}) | чистых {clean_m.mean()*100:5.1f}% "
              f"(A={a_clean:.4f})")


if __name__ == "__main__":
    main()
