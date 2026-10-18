"""Кросс-проверка артефактов под sklearn проверяющего образа (1.9.0).

Сабмит тренируется под локальным sklearn (1.5.2), а исполняется в образе
с 1.9.0. Пиклы обычно переживают это, но CalibratedClassifierCV (голова
t10) в прошлых сабмитах так не проверялся. Схема:

  python tools/check_image_compat.py --dump                 # локальный env
  .venv-image/Scripts/python.exe tools/check_image_compat.py --check

--dump  считает предсказания всех sklearn-артефактов на первых 400 строках
        трейна и пишет data/work/compat_ref.json;
--check загружает те же артефакты под другим sklearn, повторяет
        предсказания и сверяет с эталоном (допуск 1e-6).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_prep import prepare_dataframe  # noqa: E402
from src.rules import extract_rule_features, rule_prob_ok_learned  # noqa: E402
from src.text_heads import predict_all  # noqa: E402

REF = ROOT / "data" / "work" / "compat_ref.json"
N_ROWS = 400


def compute() -> dict:
    df = prepare_dataframe(ROOT / "data" / "data.csv").head(N_ROWS)
    out = {}

    rule_feats = extract_rule_features(df)
    out["rules"] = [round(float(p), 8)
                    for p in rule_prob_ok_learned(df, rule_feats)]

    for name, mapping in predict_all(df).items():
        out[name] = [round(float(mapping.get(int(i), -1.0)), 8)
                     for i in df.index]

    # e5-голова на кэшированных эмбеддингах (сам torch тут не нужен)
    emb_file = ROOT / "data" / "work" / "e5_embeddings.npy"
    if emb_file.exists():
        import joblib
        emb = np.load(emb_file)[:N_ROWS]
        for key in ("bad", "flam"):
            path = ROOT / "artifacts" / f"e5_head_{key}.joblib"
            if path.exists():
                head = joblib.load(path)
                probs = head.predict_proba(emb)[:, list(head.classes_).index(1)]
                out[f"e5_{key}"] = [round(float(p), 8) for p in probs]

    # дубликаты и kNN-артефакт: просто грузим (внутри только numpy-словарь)
    import joblib
    joblib.load(ROOT / "artifacts" / "dupes.joblib")
    knn = joblib.load(ROOT / "artifacts" / "knn_e5.joblib")
    out["knn_meta"] = [int(knn["emb"].shape[0]), int(knn["emb"].shape[1])]
    return out


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dump", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()

    import sklearn
    print(f"[compat] sklearn {sklearn.__version__}", flush=True)
    got = compute()

    if args.dump:
        REF.write_text(json.dumps(got), encoding="utf-8")
        print(f"[compat] эталон записан: {REF} ({sorted(got)})")
        return

    ref = json.loads(REF.read_text(encoding="utf-8"))
    bad = []
    for name in sorted(ref):
        if name not in got:
            bad.append(f"{name}: отсутствует в проверке")
            continue
        a, b = np.asarray(ref[name], float), np.asarray(got[name], float)
        if a.shape != b.shape or not np.allclose(a, b, atol=1e-6):
            diff = float(np.max(np.abs(a - b))) if a.shape == b.shape else -1
            bad.append(f"{name}: расхождение (max {diff})")
        else:
            print(f"[compat] {name}: бит-в-бит ок ({len(a)})")
    if bad:
        for x in bad:
            print(f"[compat] ПРОБЛЕМА: {x}")
        sys.exit(1)
    print("[compat] ВСЕ артефакты совместимы со sklearn образа")


if __name__ == "__main__":
    main()
