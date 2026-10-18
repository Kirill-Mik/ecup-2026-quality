"""Кросс-валидация: дубликаты карточек разносятся по фолдам группировкой,
иначе локальный скор врёт (утечка «та же карточка в train и val»)."""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.rules import normalize_text  # noqa: E402

N_FOLDS = 5
SEED = 42

ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "data" / "data.csv"
WORK_DIR = ROOT / "data" / "work"
FOLDS_CSV = WORK_DIR / "folds.csv"


def duplicate_group(name: str, description: str) -> str:
    key = normalize_text(name)[:200] + "|" + normalize_text(description)[:300]
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def make_folds(df: pd.DataFrame, n_folds: int = N_FOLDS, seed: int = SEED) -> pd.Series:
    """Стратификация по (category, label) с группировкой дубликатов."""
    rng = np.random.default_rng(seed)
    groups = df.apply(lambda r: duplicate_group(r.get("name", ""), r.get("description", "")), axis=1)
    df = df.assign(_group=groups)

    fold = pd.Series(-1, index=df.index)
    for (_, _), part in df.groupby(["category", "label"]):
        uniq = part["_group"].unique()
        rng.shuffle(uniq)
        gmap = {g: i % n_folds for i, g in enumerate(uniq)}
        fold.loc[part.index] = part["_group"].map(gmap)
    return fold


def load_or_make_folds(use_ocr: bool = True) -> pd.DataFrame:
    """data.csv + колонка fold; складывает data/work/folds.csv для воспроизводимости.

    Если существует data/work/ocr_train.csv (см. train/extract_ocr.py),
    распознанный текст упаковок добавляется к text той же фразой, что
    run.py в рантайме — все головы обучаются на согласованном входе.
    """
    from src.data_prep import prepare_dataframe

    df = prepare_dataframe(DATA_CSV, ROOT / "data" / "images")
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    if FOLDS_CSV.exists():
        folds = pd.read_csv(FOLDS_CSV)
        df = df.merge(folds, on="id", how="left")
        assert df["fold"].notna().all(), "folds.csv не покрывает все id"
        df["fold"] = df["fold"].astype(int)
    else:
        df["fold"] = make_folds(df).values
        df[["id", "fold"]].to_csv(FOLDS_CSV, index=False)
        print(f"[folds] создан {FOLDS_CSV}")

    ocr_path = WORK_DIR / "ocr_train.csv"
    if use_ocr and ocr_path.exists():
        ocr = pd.read_csv(ocr_path).drop_duplicates("id").set_index("id")["ocr_text"]
        matched = df["id"].map(ocr)
        m = matched.notna()
        df.loc[m, "text"] = df.loc[m, "text"] + "\nТекст с упаковки: " + matched[m].astype(str)
        print(f"[folds] подмешан OCR-текст для {int(m.sum())} товаров")
    return df


def load_pseudo() -> pd.DataFrame:
    """Псевдотрейн из data/work/pseudo.csv (см. train/pseudo_label.py):
    fold=-1 -> строки идут только в обучение, никогда в OOF-оценку.
    Пустой DataFrame, если файла нет."""
    from src.data_prep import build_text
    from src.config import norm_category

    path = WORK_DIR / "pseudo.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["category"] = df["category"].map(norm_category)
    df["text"] = build_text(df)
    df["image_paths"] = [[] for _ in range(len(df))]
    df["fold"] = -1
    print(f"[pseudo] подхвачено псевдометок: {len(df)}")
    return df


def noisy_mask(df: pd.DataFrame) -> pd.Series:
    """Строки из групп дубликатов с КОНФЛИКТОМ меток (~3.3% групп) — шум
    разметки: одинаковые карточки размечены по-разному. Исключаются из
    ОБУЧЕНИЯ (в OOF-оценке остаются — тест такой шум тоже содержит)."""
    groups = df.apply(lambda r: duplicate_group(r.get("name", ""), r.get("description", "")), axis=1)
    means = df.groupby(groups)["label"].transform("mean")
    return (means > 0) & (means < 1)
