"""kNN по мультимодальным эмбеддингам Qwen3-VL: «похожая карточка + фото».

Зачем отдельно от knn_e5. e5 читает только текст, а `imghash` ловит лишь
ПОБИТОВО совпавшую картинку (покрытие 44% БАД / 40% ЛВЖ — замер в
EXPERIMENTS.md). Между ними дыра: то же изделие, переснятое с другого ракурса
или в другой упаковке. VL-эмбеддинги такие пары сближают, а на инференсе они
и так считаются для источника `embed` — новый источник почти ничего не стоит
по времени.

Артефакт: artifacts/knn_vl.joblib (эмбеддинги трейна во float16 + метки).
"""
from typing import Dict, Optional

import joblib
import numpy as np

from src.config import ARTIFACTS_DIR, CAT_BAD, CAT_FLAM

ARTIFACT = ARTIFACTS_DIR / "knn_vl.joblib"
K_NEIGHBORS = 15
SIM_POWER = 4                    # резкий акцент на ближайших
P_CLIP = (0.02, 0.98)


def l2norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-9)


def build(train_emb: np.ndarray, labels: np.ndarray, cats: np.ndarray):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"emb": l2norm(train_emb).astype(np.float16),
                 "labels": np.asarray(labels).astype(np.int8),
                 "cats": np.asarray(cats).astype("U40")},
                ARTIFACT, compress=3)


def knn_vote(query: np.ndarray, base: np.ndarray, base_y: np.ndarray,
             k: int = K_NEIGHBORS, chunk: int = 512) -> np.ndarray:
    """P(ok) по k ближайшим соседям, вес = косинус^SIM_POWER."""
    out = np.empty(len(query), dtype=np.float64)
    kk = min(k, len(base))
    for s in range(0, len(query), chunk):
        sims = query[s:s + chunk] @ base.T
        idx = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
        rows = np.arange(len(idx))[:, None]
        w = np.clip(sims[rows, idx], 0.0, None) ** SIM_POWER
        y = base_y[idx]
        out[s:s + len(idx)] = (w * y).sum(axis=1) / np.maximum(w.sum(axis=1), 1e-9)
    return np.clip(out, *P_CLIP)


def tau_vote(query: np.ndarray, base: np.ndarray, base_y: np.ndarray,
             tau: float, chunk: int = 512):
    """P(ok) по соседям с косинусом >= tau; None там, где таких нет.

    Замер по трейну (EXPERIMENTS.md): при tau=0.98 это f1=0.997 на 48%
    ЛВЖ и f1=0.894 на 37% БАД — точнее, чем побитовый imghash, и шире.
    """
    out = np.full(len(query), np.nan)
    for s in range(0, len(query), chunk):
        sims = query[s:s + chunk] @ base.T
        hit = sims >= tau
        n = hit.sum(axis=1)
        votes = (hit * base_y).sum(axis=1)
        ok = n > 0
        idx = np.arange(s, s + len(sims))[ok]
        out[idx] = np.clip((votes[ok] + 0.5) / (n[ok] + 1.0), *P_CLIP)
    return out


def prob_ok_tau(query_emb: np.ndarray, query_cats, valid_mask,
                tau: float = 0.98) -> Dict[int, float]:
    """{позиция: P(ok)} только для строк с почти-дубликатом в трейне."""
    if not ARTIFACT.exists():
        return {}
    art = joblib.load(ARTIFACT)
    q = l2norm(query_emb)
    out: Dict[int, float] = {}
    for cat in (CAT_BAD, CAT_FLAM):
        train_m = art["cats"] == cat
        if not train_m.any():
            continue
        base = art["emb"][train_m].astype(np.float32)
        base_y = art["labels"][train_m].astype(np.float32)
        q_pos = [i for i in range(len(q)) if valid_mask[i] and query_cats[i] == cat]
        if not q_pos:
            continue
        probs = tau_vote(q[q_pos], base, base_y, tau)
        for pos, p in zip(q_pos, probs):
            if np.isfinite(p):
                out[pos] = float(p)
    return out


def prob_ok(query_emb: np.ndarray, query_cats, valid_mask,
            k: int = K_NEIGHBORS) -> Dict[int, float]:
    """{позиция строки: P(ok)} по соседям из трейна той же категории."""
    if not ARTIFACT.exists():
        return {}
    art = joblib.load(ARTIFACT)
    q = l2norm(query_emb)
    out: Dict[int, float] = {}
    for cat in (CAT_BAD, CAT_FLAM):
        train_m = art["cats"] == cat
        if not train_m.any():
            continue
        base = art["emb"][train_m].astype(np.float32)
        base_y = art["labels"][train_m].astype(np.float32)
        q_pos = [i for i in range(len(q)) if valid_mask[i] and query_cats[i] == cat]
        if not q_pos:
            continue
        probs = knn_vote(q[q_pos], base, base_y, k=k)
        for pos, p in zip(q_pos, probs):
            out[pos] = float(p)
    return out
