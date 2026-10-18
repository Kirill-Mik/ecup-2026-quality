"""Загрузка теста/трейна и подготовка признаков-«сырья»: текст + пути к картинкам."""
from pathlib import Path
from typing import List

import pandas as pd

from src.config import norm_category

_VALID_EXT = {".jpg", ".jpeg", ".png"}


def build_text(df: pd.DataFrame) -> pd.Series:
    """Единый текст карточки. NaN-описания заменяются пустой строкой."""
    name = df["name"].fillna("").astype(str)
    desc = df["description"].fillna("").astype(str)
    return "Название: " + name + "\nОписание: " + desc


def find_images_for_id(id_val, images_path: Path) -> List[str]:
    img_dir = images_path / str(id_val)
    if not img_dir.exists():
        return []
    return [
        str(img_dir / f.name)
        for f in sorted(img_dir.iterdir())
        if f.suffix.lower() in _VALID_EXT
    ]


def prepare_dataframe(data_path, images_path=None) -> pd.DataFrame:
    """Читает CSV, нормализует категории, строит text и image_paths.

    images_path=None -> ищем рядом с csv в папке images (конвенция проверяющей
    системы: путь к картинкам выводится из --test_data_path).
    """
    data_path = Path(data_path)
    if images_path is None:
        images_path = data_path.parent / "images"
    images_path = Path(images_path)

    df = pd.read_csv(data_path)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    df["category"] = df["category"].map(norm_category)
    df["text"] = build_text(df)
    if images_path.exists():
        df["image_paths"] = df["id"].apply(find_images_for_id, images_path=images_path)
    else:
        df["image_paths"] = [[] for _ in range(len(df))]
    return df
