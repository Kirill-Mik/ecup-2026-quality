"""Нейро-эмбеддинги текста (multilingual-e5-small) + kNN- и LGBM-головы.

Почему это сильно именно здесь: 41.6% трейна — дубликаты, и у теста будут
близнецы/почти-близнецы в трейне. kNN по косинусу на нейро-эмбеддингах —
обобщение мемоизации: ловит и точные копии, и перефразированные карточки.

Модель маленькая (118M, MIT): инференс 3800 карточек идёт даже на CPU
за ~1 мин, веса кладутся в архив сабмита (artifacts/models/e5-small).
"""
import os
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR, CAT_BAD, CAT_FLAM

E5_HF_ID = "intfloat/multilingual-e5-small"
E5_LOCAL_DIR = ARTIFACTS_DIR / "models" / "e5-small"
KNN_ARTIFACT = ARTIFACTS_DIR / "knn_e5.joblib"
HEAD_FILE = "e5_head_{cat_key}.joblib"
_CAT_KEYS = {CAT_BAD: "bad", CAT_FLAM: "flam"}
K_NEIGHBORS = 15


def resolve_model_path() -> Optional[str]:
    """Локальная копия в артефактах (сабмит) либо HF-кэш (обучение)."""
    if E5_LOCAL_DIR.exists():
        return str(E5_LOCAL_DIR)
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(E5_HF_ID)
    except Exception as e:  # noqa: BLE001
        print(f"[e5] модель недоступна: {e}", flush=True)
        return None


def encode_texts(texts, batch_size: int = 64, budget=None,
                 model_path: str = None) -> Optional[np.ndarray]:
    """L2-нормированные эмбеддинги (n, 384). e5 требует префикс 'query: '."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    model_path = model_path or resolve_model_path()
    if model_path is None:
        return None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(
        model_path, local_files_only=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device).eval()

    out = []
    for start in range(0, len(texts), batch_size):
        if budget is not None and not budget.has(15):
            print(f"[e5] бюджет исчерпан на {start}/{len(texts)}", flush=True)
            break
        batch = ["query: " + str(t)[:2000] for t in texts[start:start + batch_size]]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=512,
                        return_tensors="pt").to(device)
        with torch.inference_mode():
            hidden = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        emb = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        out.append(emb.float().cpu().numpy().astype(np.float32))
    del model, tokenizer
    if not out:
        return None
    res = np.vstack(out)
    if len(res) < len(texts):                      # частичный прогон -> нули
        res = np.vstack([res, np.zeros((len(texts) - len(res), res.shape[1]),
                                       dtype=np.float32)])
    return res


# ------------------------------------------------------------------ kNN
def build_knn(train_emb: np.ndarray, labels: np.ndarray, cats: np.ndarray,
              texts=None):
    """Артефакт: эмбеддинги трейна + метки (+короткие тексты для RAG-судьи)."""
    payload = {"emb": train_emb.astype(np.float32),
               "labels": labels.astype(np.int8),
               "cats": cats.astype("U40")}
    if texts is not None:
        payload["texts"] = np.array([str(t)[:280] for t in texts], dtype=object)
    joblib.dump(payload, KNN_ARTIFACT, compress=3)


def retrieve_precedents(query_emb: np.ndarray, query_cats, valid_mask,
                        k: int = 3) -> Dict[int, list]:
    """{позиция: [(текст, метка, близость), ...]} — прецеденты для RAG-судьи."""
    if not KNN_ARTIFACT.exists():
        return {}
    art = joblib.load(KNN_ARTIFACT)
    if "texts" not in art:
        return {}
    out: Dict[int, list] = {}
    for cat in (CAT_BAD, CAT_FLAM):
        train_m = art["cats"] == cat
        if not train_m.any():
            continue
        base = art["emb"][train_m]
        base_y = art["labels"][train_m]
        base_t = art["texts"][train_m]
        q_pos = [i for i in range(len(query_emb))
                 if valid_mask[i] and query_cats[i] == cat]
        if not q_pos:
            continue
        sims = query_emb[q_pos] @ base.T
        kk = min(k, base.shape[0])
        idx = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
        for row, pos in enumerate(q_pos):
            order = idx[row][np.argsort(-sims[row, idx[row]])]
            out[pos] = [(str(base_t[j]), int(base_y[j]), float(sims[row, j]))
                        for j in order]
    return out


E5_DUP_TAU = 0.97


def dup_prob_ok(query_emb: np.ndarray, query_cats, valid_mask,
                tau: float = E5_DUP_TAU) -> Dict[int, float]:
    """{позиция: P(ok)} по ПЕРЕСКАЗАННЫМ дубликатам: соседи с косинусом >= tau.

    Дополняет vlknn: там сравниваются картинки с текстом, здесь — только
    текст, поэтому ловится та же карточка с другими фото. Замер по трейну:
    БАД f1=0.895 на 42%, ЛВЖ f1=0.982 на 60%, из них 1624 строки vlknn не
    покрывает вовсе (f1 на них 0.958).
    """
    if not KNN_ARTIFACT.exists():
        return {}
    from src.vlknn import tau_vote

    art = joblib.load(KNN_ARTIFACT)
    out: Dict[int, float] = {}
    for cat in (CAT_BAD, CAT_FLAM):
        train_m = art["cats"] == cat
        if not train_m.any():
            continue
        base = art["emb"][train_m].astype(np.float32)
        base_y = art["labels"][train_m].astype(np.float32)
        q_pos = [i for i in range(len(query_emb))
                 if valid_mask[i] and query_cats[i] == cat]
        if not q_pos:
            continue
        probs = tau_vote(np.asarray(query_emb, dtype=np.float32)[q_pos],
                         base, base_y, tau)
        for pos, p in zip(q_pos, probs):
            if np.isfinite(p):
                out[pos] = float(p)
    return out


def knn_prob_ok(query_emb: np.ndarray, query_cats, valid_mask,
                k: int = K_NEIGHBORS) -> Dict[int, float]:
    """{позиция: P(ok)} — среднее меток k соседей своей категории,
    взвешенное косинусной близостью."""
    if not KNN_ARTIFACT.exists():
        return {}
    art = joblib.load(KNN_ARTIFACT)
    out: Dict[int, float] = {}
    for cat in (CAT_BAD, CAT_FLAM):
        train_m = art["cats"] == cat
        if not train_m.any():
            continue
        base = art["emb"][train_m]
        base_y = art["labels"][train_m].astype(np.float32)
        q_pos = [i for i in range(len(query_emb))
                 if valid_mask[i] and query_cats[i] == cat]
        if not q_pos:
            continue
        sims = query_emb[q_pos] @ base.T           # косинус (всё L2-нормировано)
        kk = min(k, base.shape[0])
        idx = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
        for row, pos in enumerate(q_pos):
            s = sims[row, idx[row]]
            w = np.clip(s, 0.0, None) ** 4          # резкий акцент на близких
            y = base_y[idx[row]]
            p = float((w * y).sum() / max(w.sum(), 1e-9))
            out[pos] = min(max(p, 0.02), 0.98)
    return out


# ------------------------------------------------------------------ головы
def predict_with_heads(df: pd.DataFrame, emb: np.ndarray,
                       valid_mask: np.ndarray) -> Dict[int, float]:
    """P(ok) обученной головой на e5-эмбеддингах."""
    out: Dict[int, float] = {}
    for cat in (CAT_BAD, CAT_FLAM):
        path = ARTIFACTS_DIR / HEAD_FILE.format(cat_key=_CAT_KEYS[cat])
        if not path.exists():
            continue
        head = joblib.load(path)
        mask = (df["category"] == cat).values & valid_mask
        if not mask.any():
            continue
        probs = head.predict_proba(emb[mask])[:, list(head.classes_).index(1)]
        for idx, p in zip(df.index.values[mask], probs):
            out[int(idx)] = float(p)
    return out
