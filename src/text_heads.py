"""Текстовые головы: конфигурируемый TF-IDF + классификатор, много вариантов.

Каждый обученный вариант — ОТДЕЛЬНЫЙ источник бленда (text_t3, text_t13...):
разные векторизаторы/классификаторы дают некоррелированные ошибки, и
ансамбль из 2-4 разнообразных голов стабильно сильнее одной лучшей.

Каталог конфигураций один на обучение и перебор (train/run_experiments.py).
"""
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC

from src.config import ARTIFACTS_DIR, CAT_BAD, CAT_FLAM

_CAT_KEYS = {CAT_BAD: "bad", CAT_FLAM: "flam"}
_RULES_SCALE = 10.0

# ----------------------------------------------------------- каталог вариантов
VARIANT_CONFIGS = {
    "t1": dict(desc="база: word12+char35, C=4", C=4.0),
    "t2": dict(desc="C=2", C=2.0),
    "t3": dict(desc="+правиловые признаки", C=4.0, rules=True),
    "t4": dict(desc="+правила, C=2", C=2.0, rules=True),
    "t5": dict(desc="word 1-3 +правила", C=4.0, rules=True, wng=(1, 3)),
    "t6": dict(desc="char 2-6 +правила", C=4.0, rules=True, cng=(2, 6)),
    "t7": dict(desc="char_wb->char +правила", C=4.0, rules=True, char_analyzer="char"),
    "t8": dict(desc="словари 400k/600k +правила", C=4.0, rules=True,
               mfw=400_000, mfc=600_000),
    "t9": dict(desc="min_df=1 +правила", C=4.0, rules=True, min_df_w=1, min_df_c=1),
    "t10": dict(desc="LinearSVC+калибровка +правила", clf="svc", rules=True),
    "t11": dict(desc="SGD log-loss +правила", clf="sgd", rules=True),
    "t12": dict(desc="+word-tfidf на name +правила", C=4.0, rules=True, name_vec=True),
    "t13": dict(desc="binary tf +правила", C=4.0, rules=True, binary=True),
}

# Ансамбль по умолчанию (перезаписывается автопоиском в artifacts/text_ensemble.json)
DEFAULT_ENSEMBLE = ["t3"]


class TextHead:
    """Один вариант: векторизаторы + классификатор, отдельно на категорию."""

    def __init__(self, variant: str):
        self.variant = variant
        v = VARIANT_CONFIGS[variant]
        self.cfg = v
        binary = v.get("binary", False)
        self.word_vec = TfidfVectorizer(
            analyzer="word", ngram_range=v.get("wng", (1, 2)),
            max_features=v.get("mfw", 200_000), min_df=v.get("min_df_w", 2),
            sublinear_tf=not binary, binary=binary,
        )
        self.char_vec = TfidfVectorizer(
            analyzer=v.get("char_analyzer", "char_wb"),
            ngram_range=v.get("cng", (3, 5)),
            max_features=v.get("mfc", 300_000), min_df=v.get("min_df_c", 3),
            sublinear_tf=not binary, binary=binary,
        )
        self.name_vec = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), max_features=100_000,
            min_df=2, sublinear_tf=True,
        ) if v.get("name_vec") else None
        self.clf = self._build_clf(v)

    @staticmethod
    def _build_clf(v):
        kind = v.get("clf", "logreg")
        if kind == "svc":
            return CalibratedClassifierCV(LinearSVC(C=1.0, class_weight="balanced"), cv=3)
        if kind == "sgd":
            return SGDClassifier(loss="log_loss", alpha=1e-5, class_weight="balanced",
                                 max_iter=20, random_state=42)
        return LogisticRegression(C=v.get("C", 4.0), max_iter=2000,
                                  class_weight="balanced", solver="liblinear")

    def _features(self, df: pd.DataFrame, fit: bool, rule_feats):
        texts = df["text"].fillna("").tolist()
        if fit:
            parts = [self.word_vec.fit_transform(texts), self.char_vec.fit_transform(texts)]
            if self.name_vec is not None:
                parts.append(self.name_vec.fit_transform(df["name"].fillna("")))
        else:
            parts = [self.word_vec.transform(texts), self.char_vec.transform(texts)]
            if self.name_vec is not None:
                parts.append(self.name_vec.transform(df["name"].fillna("")))
        if self.cfg.get("rules"):
            parts.append(csr_matrix(rule_feats / _RULES_SCALE))
        return hstack(parts).tocsr()

    def fit(self, df: pd.DataFrame, rule_feats: Optional[np.ndarray]):
        x = self._features(df, fit=True, rule_feats=rule_feats)
        self.clf.fit(x, df["label"].values)
        return self

    def predict_proba_ok(self, df: pd.DataFrame, rule_feats) -> np.ndarray:
        x = self._features(df, fit=False, rule_feats=rule_feats)
        return self.clf.predict_proba(x)[:, list(self.clf.classes_).index(1)]

    # ------------------------------------------------------------- io
    def save(self, cat: str, art_dir: Path = ARTIFACTS_DIR, suffix: str = ""):
        art_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, art_dir / f"text_{self.variant}_{_CAT_KEYS[cat]}{suffix}.joblib")

    @staticmethod
    def load_all(variant: str, cat: str, art_dir: Path = ARTIFACTS_DIR) -> list:
        """Финальная голова + фолдовые (если сохранялись с --save-folds)."""
        heads = []
        base = art_dir / f"text_{variant}_{_CAT_KEYS[cat]}.joblib"
        if base.exists():
            heads.append(joblib.load(base))
        for p in sorted(art_dir.glob(f"text_{variant}_{_CAT_KEYS[cat]}_f*.joblib")):
            heads.append(joblib.load(p))
        return heads


def ensemble_variants(art_dir: Path = ARTIFACTS_DIR) -> list:
    """Список вариантов ансамбля: artifacts/text_ensemble.json либо дефолт."""
    import json
    path = art_dir / "text_ensemble.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return list(DEFAULT_ENSEMBLE)


def predict_all(df: pd.DataFrame, text_col: str = "text") -> Dict[str, Dict[int, float]]:
    """{"text_t3": {idx: P(ok)}, ...} по всем головам ансамбля из артефактов."""
    from src.rules import extract_rule_features

    sub = df.copy()
    sub["text"] = df[text_col]
    rule_feats = extract_rule_features(sub, text_col="text")

    out: Dict[str, Dict[int, float]] = {}
    for variant in ensemble_variants():
        source: Dict[int, float] = {}
        for cat in (CAT_BAD, CAT_FLAM):
            mask = (sub["category"] == cat).values
            if not mask.any():
                continue
            heads = TextHead.load_all(variant, cat)
            if not heads:
                continue
            # усреднение логитов финальной и фолдовых голов
            import math
            acc = np.zeros(int(mask.sum()))
            for head in heads:
                p = np.clip(head.predict_proba_ok(sub[mask], rule_feats[mask]),
                            1e-6, 1 - 1e-6)
                acc += np.log(p / (1 - p))
            probs = 1.0 / (1.0 + np.exp(-acc / len(heads)))
            for idx, p in zip(sub.index[mask], probs):
                source[int(idx)] = float(p)
        if source:
            out[f"text_{variant}"] = source
    return out
