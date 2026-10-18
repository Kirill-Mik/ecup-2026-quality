"""e5-эмбеддинги трейна + kNN-голова + логрег-голова + tau-дубликаты. CPU.

Отличия от ecup2-quality/train/train_e5.py:
  * голова ВСЕГДА логрег: lightgbm в проверяющем образе отсутствует, артефакт
    с LGBM на сервере молча мёртв (урок сабмита №1);
  * добавлен честный OOF источника e5dup (tau-соседи ТОЛЬКО из чужих фолдов);
  * kNN-артефакт без текстов прецедентов (RAG-судья исключён из сабмита).

Выход: artifacts/knn_e5.joblib, artifacts/e5_head_{cat}.joblib,
OOF: oof_knn.csv, oof_e5.csv, oof_e5dup.csv.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sklearn.linear_model import LogisticRegression  # noqa: E402

from src.config import ARTIFACTS_DIR, CAT_BAD, CAT_FLAM  # noqa: E402
from src.emb_text import (E5_DUP_TAU, E5_LOCAL_DIR, HEAD_FILE, K_NEIGHBORS,  # noqa: E402
                          _CAT_KEYS, build_knn, encode_texts)
from src.vlknn import tau_vote  # noqa: E402
from train.cv_utils import N_FOLDS, WORK_DIR, load_or_make_folds, noisy_mask  # noqa: E402
from train.metric import report  # noqa: E402

EMB_FILE = WORK_DIR / "e5_embeddings.npy"


def make_head():
    # НИКАКОГО lightgbm: его нет в проверяющем образе (odsai/ecup26-quality-
    # baseline:1.0), и артефакт с ним просто не загрузится на сервере.
    return LogisticRegression(C=4.0, max_iter=3000, class_weight="balanced")


def knn_oof(emb, y, cats, folds, k=K_NEIGHBORS):
    """OOF kNN: база — только чужие фолды."""
    oof = np.full(len(y), np.nan)
    for f in range(N_FOLDS):
        tr, va = folds != f, folds == f
        for cat in (CAT_BAD, CAT_FLAM):
            base_m = tr & (cats == cat)
            q_m = va & (cats == cat)
            if not base_m.any() or not q_m.any():
                continue
            sims = emb[q_m] @ emb[base_m].T
            kk = min(k, int(base_m.sum()))
            idx = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
            base_y = y[base_m].astype(np.float32)
            rows = np.where(q_m)[0]
            for r, gi in enumerate(rows):
                s = sims[r, idx[r]]
                w = np.clip(s, 0.0, None) ** 4
                oof[gi] = float((w * base_y[idx[r]]).sum() / max(w.sum(), 1e-9))
    return np.clip(oof, 0.02, 0.98)


def e5dup_oof(emb, y, cats, folds, tau=E5_DUP_TAU):
    """OOF tau-дубликатов: соседи с косинусом >= tau ТОЛЬКО из чужих фолдов.
    NaN там, где таких соседей нет (узкий, но точный источник)."""
    oof = np.full(len(y), np.nan)
    for f in range(N_FOLDS):
        tr, va = folds != f, folds == f
        for cat in (CAT_BAD, CAT_FLAM):
            base_m = tr & (cats == cat)
            q_m = va & (cats == cat)
            if not base_m.any() or not q_m.any():
                continue
            probs = tau_vote(emb[q_m].astype(np.float32),
                             emb[base_m].astype(np.float32),
                             y[base_m].astype(np.float32), tau)
            rows = np.where(q_m)[0]
            for r, gi in enumerate(rows):
                if np.isfinite(probs[r]):
                    oof[gi] = probs[r]
    return oof


def main():
    df = load_or_make_folds(use_ocr=False)
    y = df["label"].values
    cats = df["category"].values
    folds = df["fold"].values

    # ---------------------------------------------- эмбеддинги (кэшируются)
    if EMB_FILE.exists():
        emb = np.load(EMB_FILE)
        print(f"[e5] эмбеддинги из кэша: {emb.shape}")
        assert len(emb) == len(df), "кэш эмбеддингов не совпал по длине с data.csv"
    else:
        if not E5_LOCAL_DIR.exists():
            sys.exit(f"[e5] нет модели {E5_LOCAL_DIR} и нет кэша эмбеддингов")
        emb = encode_texts(df["text"].tolist(), model_path=str(E5_LOCAL_DIR))
        if emb is None:
            sys.exit("[e5] извлечение не удалось")
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        np.save(EMB_FILE, emb)
        print(f"[e5] эмбеддинги посчитаны: {emb.shape}")

    # ---------------------------------------------- kNN
    oof_k = knn_oof(emb, y, cats, folds)
    pd.DataFrame({"id": df["id"], "oof_knn": oof_k}).to_csv(
        WORK_DIR / "oof_knn.csv", index=False)
    report(y, (oof_k >= 0.5).astype(int), cats, tag="knn-e5 oof@0.5")
    build_knn(emb, y, cats)                      # без текстов: RAG исключён
    print("[e5] kNN-артефакт сохранён")

    # ---------------------------------------------- e5dup (tau-дубликаты)
    oof_d = e5dup_oof(emb, y, cats, folds)
    covered = np.isfinite(oof_d)
    pd.DataFrame({"id": df["id"], "oof_e5dup": oof_d}).to_csv(
        WORK_DIR / "oof_e5dup.csv", index=False)
    if covered.any():
        acc = ((oof_d[covered] >= 0.5).astype(int) == y[covered]).mean()
        print(f"[e5dup] OOF: покрытие {covered.mean()*100:.1f}%, "
              f"точность {acc*100:.1f}%")

    # ---------------------------------------------- логрег-голова
    clean = ~noisy_mask(df).values
    oof_h = pd.Series(np.nan, index=df.index)
    for cat in (CAT_BAD, CAT_FLAM):
        m = (cats == cat)
        for f in range(N_FOLDS):
            tr = m & (folds != f) & clean
            va = m & (folds == f)
            head = make_head()
            head.fit(emb[tr], y[tr])
            oof_h.iloc[np.where(va)[0]] = head.predict_proba(emb[va])[:, list(head.classes_).index(1)]
        final = make_head()
        final.fit(emb[m & clean], y[m & clean])
        joblib.dump(final, ARTIFACTS_DIR / HEAD_FILE.format(cat_key=_CAT_KEYS[cat]))
    pd.DataFrame({"id": df["id"], "oof_e5": oof_h.values}).to_csv(
        WORK_DIR / "oof_e5.csv", index=False)
    report(y, (oof_h.values >= 0.5).astype(int), cats, tag="e5-head oof@0.5")


if __name__ == "__main__":
    main()
