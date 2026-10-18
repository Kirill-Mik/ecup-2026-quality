"""Дешёвые головы: обученные правила + мемоизация дубликатов. CPU, <1 мин.

  python train/train_cheap_heads.py

Выход:
  artifacts/rules_head_{bad,flam}.joblib  — логрег на 11 правиловых признаках
  artifacts/dupes.joblib                  — хэш-карты дубликатов трейна
  data/work/oof_rules.csv                 — OOF P(ok) правиловой головы
  data/work/oof_dupes.csv                 — OOF P(ok) name-хэша (кросс-фолд)
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.config import ARTIFACTS_DIR, CAT_BAD, CAT_FLAM  # noqa: E402
from src.dupes import ALPHA, P_CLIP_NAME, build_maps, name_key, save_maps  # noqa: E402
from src.rules import extract_rule_features  # noqa: E402
from train.cv_utils import N_FOLDS, WORK_DIR, load_or_make_folds  # noqa: E402
from train.metric import report  # noqa: E402


def make_head():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"),
    )


def main():
    df = load_or_make_folds()
    feats = extract_rule_features(df)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------ правиловая голова
    oof = pd.Series(np.nan, index=df.index)
    for cat, key in ((CAT_BAD, "bad"), (CAT_FLAM, "flam")):
        m = (df["category"] == cat).values
        for fold in range(N_FOLDS):
            tr = m & (df["fold"] != fold).values
            va = m & (df["fold"] == fold).values
            head = make_head().fit(feats[tr], df["label"].values[tr])
            oof.iloc[np.where(va)[0]] = head.predict_proba(feats[va])[:, list(head.classes_).index(1)]
        final = make_head().fit(feats[m], df["label"].values[m])
        joblib.dump(final, ARTIFACTS_DIR / f"rules_head_{key}.joblib")
    pd.DataFrame({"id": df["id"], "oof_rules": oof.values}).to_csv(
        WORK_DIR / "oof_rules.csv", index=False)
    report(df["label"].values, (oof.values >= 0.5).astype(int),
           df["category"].values, tag="rules-head oof@0.5")

    # ------------------------------------------------ дубликаты
    save_maps(build_maps(df))
    print(f"[dupes] карты сохранены: {ARTIFACTS_DIR / 'dupes.joblib'}")

    # OOF для name-хэша: карта из чужих фолдов -> попадания в своём
    oof_dup = pd.Series(np.nan, index=df.index)
    keys = df["name"].fillna("").map(name_key)
    for fold in range(N_FOLDS):
        tr = df["fold"] != fold
        agg = pd.DataFrame({"k": keys[tr], "y": df.loc[tr, "label"]}).groupby("k")["y"].agg(["sum", "count"])
        p_map = ((agg["sum"] + ALPHA) / (agg["count"] + 2 * ALPHA)).clip(*P_CLIP_NAME)
        hits = keys[df["fold"] == fold].map(p_map)
        oof_dup.loc[hits.index] = hits
    covered = oof_dup.notna()
    pd.DataFrame({"id": df["id"], "oof_dup_name": oof_dup.values}).to_csv(
        WORK_DIR / "oof_dupes.csv", index=False)
    if covered.any():
        acc = ((oof_dup[covered] >= 0.5).astype(int) == df.loc[covered, "label"]).mean()
        print(f"[dupes] name-хэш OOF: покрытие {covered.mean()*100:.1f}%, точность {acc*100:.1f}%")


if __name__ == "__main__":
    main()
