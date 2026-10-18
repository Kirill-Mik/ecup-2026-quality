"""Обучение ансамбля текстовых голов: каждый вариант -> артефакты + OOF.

  python train/train_text_heads.py --variants t3 t9 t13
  python train/train_text_heads.py                # из artifacts/text_ensemble.json

Строки с конфликтом меток внутри групп дубликатов (шум разметки)
исключаются из обучения (--keep-noisy отключает).
Выход: artifacts/text_{v}_{cat}.joblib, artifacts/text_ensemble.json,
data/work/oof_text_{v}.csv.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import ARTIFACTS_DIR, CAT_BAD, CAT_FLAM  # noqa: E402
from src.rules import extract_rule_features  # noqa: E402
from src.text_heads import VARIANT_CONFIGS, TextHead, ensemble_variants  # noqa: E402
from train.cv_utils import N_FOLDS, WORK_DIR, load_or_make_folds, noisy_mask  # noqa: E402
from train.metric import report  # noqa: E402


def train_variant(variant: str, df, rule_feats, clean_mask,
                  save_folds: bool = False) -> pd.Series:
    oof = pd.Series(np.nan, index=df.index)
    for cat in (CAT_BAD, CAT_FLAM):
        cat_m = (df["category"] == cat).values
        for fold in range(N_FOLDS):
            tr_m = cat_m & (df["fold"] != fold).values & clean_mask
            va_m = cat_m & (df["fold"] == fold).values
            head = TextHead(variant).fit(df[tr_m], rule_feats[tr_m])
            oof.iloc[np.where(va_m)[0]] = head.predict_proba_ok(df[va_m], rule_feats[va_m])
            if save_folds:
                head.save(cat, suffix=f"_f{fold}")
        final_m = cat_m & clean_mask
        TextHead(variant).fit(df[final_m], rule_feats[final_m]).save(cat)
        print(f"[text] {variant} {cat}: обучен и сохранён")
    return oof


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="*", default=None,
                        help=f"из {list(VARIANT_CONFIGS)}; дефолт — text_ensemble.json")
    parser.add_argument("--keep-noisy", action="store_true",
                        help="не выбрасывать конфликтные дубликаты из обучения")
    parser.add_argument("--pseudo", action="store_true",
                        help="добавить псевдометки теста (data/work/pseudo.csv)")
    parser.add_argument("--save-folds", action="store_true",
                        help="сохранять фолдовые головы: инференс усреднит 6 голов")
    args = parser.parse_args()

    df = load_or_make_folds()
    if args.keep_noisy:
        clean_mask = np.ones(len(df), dtype=bool)
    else:
        noisy = noisy_mask(df).values
        clean_mask = ~noisy
        print(f"[text] исключено из обучения шумных строк: {noisy.sum()} "
              f"({noisy.mean()*100:.1f}%)")
    if args.pseudo:
        from train.cv_utils import load_pseudo
        ps = load_pseudo()
        if len(ps):
            df = pd.concat([df, ps], ignore_index=True)
            clean_mask = np.concatenate([clean_mask, np.ones(len(ps), dtype=bool)])
    rule_feats = extract_rule_features(df)

    variants = args.variants or ensemble_variants()
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    real = (df["fold"] >= 0).values          # без псевдострок (fold=-1)
    for v in variants:
        oof = train_variant(v, df, rule_feats, clean_mask,
                            save_folds=args.save_folds)
        pd.DataFrame({"id": df["id"][real],
                      f"oof_text_{v}": oof.values[real]}).to_csv(
            WORK_DIR / f"oof_text_{v}.csv", index=False)
        report(df["label"].values[real], (oof.values[real] >= 0.5).astype(int),
               df["category"].values[real], tag=f"text_{v} oof@0.5")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "text_ensemble.json").write_text(
        json.dumps(list(variants), ensure_ascii=False), encoding="utf-8")
    print(f"[text] ансамбль: {list(variants)} -> artifacts/text_ensemble.json")


if __name__ == "__main__":
    main()
