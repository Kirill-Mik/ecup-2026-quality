"""Подбор весов и порогов ПО ПРОФИЛЯМ ДЕГРАДАЦИИ + коридоры доли «не бан».

  python train/tune_blend.py

Ключевые отличия от старого tune_thresholds.py:
  * формула бленда и дефолт веса (0 для неуказанного источника) здесь и в
    src/blend.py ОДНА И ТА ЖЕ. В старом коде тюнер считал с дефолтом 0,
    а run.py — с дефолтом 1: дома оценивалась одна модель, на сервер
    ехала другая;
  * помимо полного состава тюнится профиль no_e5 (вдруг на сервере
    отвалится torch/эмбеддинги) — у него СВОИ веса и СВОИ пороги;
  * для каждого профиля и категории пишется коридор доли «не бан» на OOF —
    run.py заменит порог квантильным, если доля вышла за коридор
    (страховка от «ЛВЖ схлопнулась в всегда-бан»);
  * никакого стекера: меньше кода = меньше поверхность отказа, а выигрыш
    стекера в прошлых замерах не превышал 0.002.

Выход: artifacts/blend.json, artifacts/thresholds.json (справочно),
data/work/oof_report.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.blend import _logit, _sigmoid  # noqa: E402
from src.config import ARTIFACTS_DIR, CAT_BAD, CAT_FLAM  # noqa: E402
from train.cv_utils import WORK_DIR, load_or_make_folds  # noqa: E402
from train.metric import metric_a, report  # noqa: E402

N_BOOTSTRAP = 51
SEED = 42
WEIGHT_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

E5_FAMILY = ("knn", "e5", "e5dup")
PROFILES = [
    ("full", None),                 # все источники
    ("no_e5", E5_FAMILY),           # исключить e5-семейство
]


# ------------------------------------------------------------------ пороги
def exact_best_threshold(y_true: np.ndarray, prob_ok: np.ndarray):
    """Порог, максимизирующий macro-F1 двух классов внутри категории."""
    order = np.argsort(prob_ok)
    y_sorted = y_true[order]
    p_sorted = prob_ok[order]
    n = len(y_sorted)
    total_ok = int(y_sorted.sum())
    total_ban = n - total_ok

    cum_ok = np.concatenate([[0], np.cumsum(y_sorted)])
    ks = np.arange(n + 1)
    banned_ok = cum_ok
    banned_ban = ks - banned_ok
    kept_ok = total_ok - banned_ok
    f1_ban = np.where(ks + total_ban > 0, 2 * banned_ban / np.maximum(ks + total_ban, 1), 0.0)
    f1_ok = np.where((n - ks) + total_ok > 0, 2 * kept_ok / np.maximum((n - ks) + total_ok, 1), 0.0)
    macro = (f1_ban + f1_ok) / 2
    k = int(np.argmax(macro))
    if k == 0:
        thr = p_sorted[0] / 2
    elif k == n:
        thr = (p_sorted[-1] + 1.0) / 2
    else:
        thr = (p_sorted[k - 1] + p_sorted[k]) / 2
    return float(thr), float(macro[k])


def bootstrap_threshold(y_true, prob_ok, n_boot=N_BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    thrs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        thrs.append(exact_best_threshold(y_true[idx], prob_ok[idx])[0])
    return float(np.median(thrs))


# ------------------------------------------------------------------ источники
def load_oof(df: pd.DataFrame) -> dict:
    """Все OOF-файлы рабочей папки: имя источника -> массив P(ok)/NaN."""
    sources = {}

    def add(name, path, col):
        merged = df[["id"]].merge(pd.read_csv(path), on="id", how="left")
        sources[name] = merged[col].values.astype(float)
        print(f"[tune] источник {name}: {np.isfinite(sources[name]).sum()} скоров")

    for path in sorted(WORK_DIR.glob("oof_text_*.csv")):
        v = path.stem.replace("oof_text_", "")
        add(f"text_{v}", path, f"oof_text_{v}")
    fixed = [("rules", "oof_rules.csv", "oof_rules"),
             ("dup_name", "oof_dupes.csv", "oof_dup_name"),
             ("knn", "oof_knn.csv", "oof_knn"),
             ("e5", "oof_e5.csv", "oof_e5"),
             ("e5dup", "oof_e5dup.csv", "oof_e5dup")]
    for name, fname, col in fixed:
        p = WORK_DIR / fname
        if p.exists():
            add(name, p, col)
    return sources


# ------------------------------------------------------------------ бленд
def blend_arrays(cat_sources: dict, weights: dict, n: int) -> np.ndarray:
    """ТА ЖЕ формула, что src.blend.blend_with_profile: вес по умолчанию 0."""
    num = np.zeros(n)
    den = np.zeros(n)
    for name, arr in cat_sources.items():
        w = float(weights.get(name, 0.0))
        if w <= 0:
            continue
        mask = np.isfinite(arr)
        logits = np.array([_logit(p) for p in np.where(mask, arr, 0.5)])
        num += np.where(mask, w * logits, 0.0)
        den += np.where(mask, w, 0.0)
    with np.errstate(invalid="ignore"):
        out = np.where(den > 0, num / np.maximum(den, 1e-9), 0.0)
    return np.array([_sigmoid(x) for x in out])


def _descend(cat_sources: dict, y: np.ndarray, n: int, init: dict,
             passes: int = 4):
    weights = {name: float(init.get(name, 0.0)) for name in cat_sources}

    def score(w):
        probs = blend_arrays(cat_sources, w, n)
        return exact_best_threshold(y, probs)[1], probs

    best_f1, best_probs = score(weights)
    for _ in range(passes):
        improved = False
        for name in list(weights):
            for cand_w in WEIGHT_GRID:
                if cand_w == weights[name]:
                    continue
                cand = dict(weights, **{name: cand_w})
                if all(v == 0 for v in cand.values()):
                    continue
                f1, probs = score(cand)
                if f1 > best_f1 + 1e-5:
                    best_f1, best_probs, weights = f1, probs, cand
                    improved = True
        if not improved:
            break
    return weights, best_f1, best_probs


def tune_weights(cat_sources: dict, y: np.ndarray, n: int, passes: int = 4,
                 warm_starts: list = None):
    """Координатный спуск от нескольких стартов; берём лучший финиш.

    Жадный спуск от «все по 1.0» застревает в локальных оптимумах, когда
    слабые источники в старте тянут бленд вниз. Тёплые старты (например,
    веса профиля без e5) дают спуску шанс НЕ включать слабое семейство.
    """
    starts = [{name: 1.0 for name in cat_sources}]
    for ws in warm_starts or []:
        starts.append({name: float(ws.get(name, 0.0)) for name in cat_sources})
    best = None
    for init in starts:
        weights, f1, probs = _descend(cat_sources, y, n, init, passes)
        if best is None or f1 > best[1] + 1e-5:
            best = (weights, f1, probs)
    return best


def rate_band(rate: float) -> dict:
    """Коридор допустимой доли «не бан» вокруг OOF-доли.

    Для редкого класса (ЛВЖ, ~4-5%) — мультипликативный x/÷3: сдвиги
    сэмплирования публичного теста легко удваивают долю. Для частого
    (БАД, ~70%+) — аддитивный +-0.2/0.25.
    """
    if rate < 0.2:
        return {"target": rate, "lo": rate / 3.0, "hi": min(0.5, rate * 3.0)}
    return {"target": rate,
            "lo": max(0.05, rate - 0.25),
            "hi": min(0.98, rate + 0.20)}


# ------------------------------------------------------------------ main
def main():
    df = load_or_make_folds(use_ocr=False)
    y_all = df["label"].values
    cats = df["category"].values
    sources_all = load_oof(df)

    expected = {"text_t3", "text_t5", "text_t7", "text_t9", "text_t10",
                "text_t12", "rules", "dup_name", "knn", "e5", "e5dup"}
    missing = expected - set(sources_all)
    if missing:
        sys.exit(f"[tune] НЕ ХВАТАЕТ OOF-источников: {sorted(missing)} — "
                 f"сначала дообучите головы")

    profiles_out = []
    report_out = {}
    no_e5_weights = {}          # тёплый старт для полного профиля
    for prof_name, exclude in sorted(PROFILES, key=lambda p: p[0] != "no_e5"):
        sources = {k: v for k, v in sources_all.items()
                   if not exclude or k not in exclude}
        prof_weights, prof_thrs, prof_rates = {}, {}, {}
        all_probs = np.zeros(len(df))
        for cat in (CAT_BAD, CAT_FLAM):
            m = cats == cat
            y = y_all[m]
            cat_sources = {k: v[m] for k, v in sources.items()}
            warm = [no_e5_weights[cat]] if cat in no_e5_weights else []
            w, _, probs = tune_weights(cat_sources, y, int(m.sum()),
                                       warm_starts=warm)
            if prof_name == "no_e5":
                no_e5_weights[cat] = w
            thr = bootstrap_threshold(y, probs)
            preds = (probs >= thr).astype(int)
            from sklearn.metrics import f1_score
            f1b = f1_score(y, preds, average="macro")
            rate = float(preds.mean())
            prof_weights[cat] = {k: v for k, v in w.items()}
            prof_thrs[cat] = thr
            prof_rates[cat] = rate_band(rate)
            all_probs[m] = probs
            print(f"[tune] {prof_name}/{cat}: f1={f1b:.4f} thr={thr:.4f} "
                  f"ok-rate={rate:.3f} веса={ {k: v for k, v in w.items() if v > 0} }")

        preds_all = np.array([
            int(all_probs[i] >= prof_thrs[cats[i]]) for i in range(len(df))])
        a = metric_a(y_all, preds_all, cats)
        report(y_all, preds_all, cats, tag=f"{prof_name} oof")

        requires = []
        if not exclude:
            requires = [s for s in E5_FAMILY
                        if any(prof_weights[c].get(s, 0) > 0
                               for c in (CAT_BAD, CAT_FLAM))]
        profiles_out.append({
            "name": prof_name,
            "requires": requires,
            "weights": prof_weights,
            "thresholds": prof_thrs,
            "pos_rate": prof_rates,
            "oof_a": round(a, 4),
        })
        report_out[prof_name] = {
            "oof_a": round(a, 4),
            "thresholds": prof_thrs,
            "pos_rate": prof_rates,
        }

    # порядок профилей в конфиге = порядок предпочтения в рантайме
    canon = [name for name, _ in PROFILES]
    profiles_out.sort(key=lambda p: canon.index(p["name"]))
    cfg = {
        "source_order": sorted(sources_all),
        "dup_exact_weight": 3.0,        # согласие меток в группах дублей 96.7%
        "profiles": profiles_out,
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "blend.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    (ARTIFACTS_DIR / "thresholds.json").write_text(
        json.dumps(profiles_out[0]["thresholds"], ensure_ascii=False, indent=2),
        encoding="utf-8")
    (WORK_DIR / "oof_report.json").write_text(
        json.dumps(report_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[tune] blend.json (профили) + thresholds.json + oof_report.json записаны")


if __name__ == "__main__":
    main()
