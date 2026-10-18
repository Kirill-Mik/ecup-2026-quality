"""E-CUP 2026, задача 2 «Контроль качества» — точка входа инференса.

Выводы из сабмитов №1 (public 0.71) и №2 (0.67) при домашней оценке 0.92:
терялись мы не на модели, а на доставке — на сервере молча отваливались
шаги, которые дома никто не прогонял, а пороги оставались от полного
состава. Отсюда четыре правила этой версии:

 1. В каскаде НЕТ шагов, которые нельзя прогнать в репетиции дома: только
    CPU-совместимые головы, все веса моделей — в архиве. Ничего из
    /shared_models, никакого OCR/судьи/VL «оживающих только на сервере».
 2. Валидный submit.csv записывается через секунды после старта и затем
    только улучшается. Любое падение дальше оставляет рабочий файл.
 3. Бленд и пороги подобраны тюнером с теми же формулами и дефолтами
    (вес неуказанного источника = 0). Если источник не посчитался,
    включается профиль, чьи веса И ПОРОГИ подобраны на OOF именно без
    него, — а не пороги полного состава.
 4. Страховка от схлопывания категории: если доля «не бан» вышла за
    коридор OOF, порог заменяется квантильным (rate_guard).

Каскад: правила -> дубликаты -> [запись] -> TF-IDF-головы -> [запись]
        -> e5 (kNN + голова + tau-дубликаты) -> бленд по профилю
        -> rate_guard -> [финальная запись].
"""
# Обе переменные аллокатора: PYTORCH_CUDA_ALLOC_CONF читается torch <= 2.7,
# PYTORCH_ALLOC_CONF — torch >= 2.8. Одна из них молча игнорируется — и
# ставившая только новую версия молча теряла настройку (урок v3).
import os as _os
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
_os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from src.blend import (apply_thresholds, blend_with_profile, load_deploy,
                       rate_guard, select_profile)
from src.comments import build_results, template_comments
from src.data_prep import prepare_dataframe
from src.rules import extract_rule_features, rule_prob_ok_learned
from src.text_heads import predict_all as text_predict_all
from src.timebudget import TimeBudget
from src.validate import validate_submission

# Головы, которые ОБЯЗАНЫ посчитаться (иначе это уже не наш сабмит);
# e5-семейство может деградировать — на этот случай есть профиль no_e5.
CORE_SOURCES = {"rules", "text_t3", "text_t5", "text_t7", "text_t9",
                "text_t10", "text_t12"}


def parse_args():
    p = argparse.ArgumentParser(description="E-CUP 2026 quality control cascade")
    p.add_argument("--test_data_path", "--test-data-path", "-i",
                   dest="test_data_path", required=True)
    p.add_argument("--output_path", "--output-path", "-o",
                   dest="output_path", required=True)
    return p.parse_args()


def write_submission(df: pd.DataFrame, preds: np.ndarray, comments, output_path: str):
    """Единственное место записи: собирает result, валидирует, пишет."""
    results = build_results(df, preds, comments)
    out = pd.DataFrame({"id": df["id"].values, "result": results})
    errors = validate_submission(out, expected_ids=df["id"].tolist())
    if errors:
        print(f"[write] ВАЛИДАЦИЯ ПРОВАЛЕНА: {errors[:5]}", flush=True)
        # чинимся шаблонами — формат важнее красоты
        results = build_results(df, preds, template_comments(df, preds))
        out = pd.DataFrame({"id": df["id"].values, "result": results})
    out.to_csv(output_path, index=False)
    print(f"[write] {output_path}: {len(out)} строк", flush=True)


def blend_and_write(df, sources, cfg, output_path, final: bool = False):
    """Бленд по профилю от фактического состава -> пороги -> запись."""
    present = {k for k, v in sources.items() if v}
    profile = select_profile(cfg, present)
    probs = blend_with_profile(df, sources, profile,
                               float(cfg.get("dup_exact_weight", 3.0)))
    preds = apply_thresholds(df, probs, profile.get("thresholds"))
    guard_notes = []
    if final:
        preds, guard_notes = rate_guard(df, probs, preds, profile)
        for note in guard_notes:
            print(f"[guard] {note}", flush=True)
    write_submission(df, preds, template_comments(df, preds), output_path)
    return profile, probs, preds, guard_notes


def main() -> None:
    args = parse_args()

    df = prepare_dataframe(Path(args.test_data_path))
    budget = TimeBudget(len(df))
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    except Exception:  # noqa: BLE001
        gpu = "no torch"
    budget.log(f"start, n={len(df)}, device={gpu}")

    # -------------------------------------------------- манифест репетиции
    try:
        from src.manifest import load_manifest, verify_manifest
        man = load_manifest()
        problems = verify_manifest(check_hashes=len(df) > 300)
        if problems:
            for p in problems[:10]:
                print(f"[manifest] ПРОБЛЕМА: {p}", flush=True)
        else:
            print(f"[manifest] ок: репетиция {man.get('rehearsed_at')}, "
                  f"метрика реп.={man.get('rehearsal_metric_a')}, "
                  f"OOF={man.get('oof_metric_a')}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[manifest] проверка не удалась: {e}", flush=True)

    cfg = load_deploy()
    sources = {}
    failed = []

    # -------------------------------------------------- шаг 0: правила
    # Без try: правила — чистый numpy+sklearn по артефакту из архива.
    # Если упало здесь, значит окружение сломано целиком — пусть падает
    # громко, шаблонный фолбэк ниже всё равно запишет валидный файл.
    rule_feats = extract_rule_features(df)
    rules_probs = rule_prob_ok_learned(df, rule_feats)
    sources["rules"] = {int(idx): float(p)
                       for idx, p in zip(df.index, rules_probs)}
    budget.log("rules")

    # -------------------------------------------------- шаг 0б: дубликаты
    try:
        from src.dupes import predict as dupes_predict
        dup_exact, dup_name = dupes_predict(df)
        if dup_exact:
            sources["dup_exact"] = dup_exact
        if dup_name:
            sources["dup_name"] = dup_name
        print(f"[dupes] exact={len(dup_exact)} name={len(dup_name)}", flush=True)
    except Exception as e:  # noqa: BLE001
        failed.append(("dupes", repr(e)))
        print(f"[dupes] УПАЛ: {e}", flush=True)

    # -------------------------------------------------- запись №1 (страховка)
    blend_and_write(df, sources, cfg, args.output_path)
    budget.log("checkpoint 1 (rules+dupes)")

    # -------------------------------------------------- шаг 1: TF-IDF-головы
    try:
        text_sources = text_predict_all(df)
        sources.update(text_sources)
        print(f"[text] головы: {sorted(text_sources)}", flush=True)
    except Exception as e:  # noqa: BLE001
        failed.append(("text", repr(e)))
        print(f"[text] УПАЛ: {e}", flush=True)
        traceback.print_exc()

    blend_and_write(df, sources, cfg, args.output_path)
    budget.log("checkpoint 2 (+text)")

    # -------------------------------------------------- шаг 2: e5-семейство
    try:
        if budget.has(45):
            from src.emb_text import (dup_prob_ok, encode_texts, knn_prob_ok,
                                      predict_with_heads)
            emb = encode_texts(df["text"].fillna("").tolist(), budget=budget)
            if emb is None:
                failed.append(("e5", "encode_texts вернул None"))
            else:
                valid = np.abs(emb).sum(axis=1) > 0
                cats_arr = df["category"].values
                knn = knn_prob_ok(emb, cats_arr, valid)
                if knn:
                    sources["knn"] = {int(df.index[pos]): p
                                      for pos, p in knn.items()}
                e5_src = predict_with_heads(df, emb, valid)
                if e5_src:
                    sources["e5"] = e5_src
                dup = dup_prob_ok(emb, cats_arr, valid)
                if dup:
                    sources["e5dup"] = {int(df.index[pos]): p
                                        for pos, p in dup.items()}
                print(f"[e5] knn={len(knn)} head={len(e5_src)} dup={len(dup)} "
                      f"строк с эмбеддингом={int(valid.sum())}/{len(df)}",
                      flush=True)
        else:
            print("[e5] нет бюджета — профиль no_e5", flush=True)
    except Exception as e:  # noqa: BLE001
        failed.append(("e5", repr(e)))
        print(f"[e5] УПАЛ: {e}", flush=True)
        traceback.print_exc()
    budget.log("e5")

    # -------------------------------------------------- финал: бленд+пороги
    profile, probs, preds, guard_notes = blend_and_write(
        df, sources, cfg, args.output_path, final=True)
    budget.log("final")

    # -------------------------------------------------- сводка для репетиции
    present = sorted(k for k, v in sources.items() if v)
    missing_core = sorted(CORE_SOURCES - set(present))
    cats = df["category"].values
    rates = {cat: round(float(preds[cats == cat].mean()), 4)
             for cat in sorted(set(cats))}
    summary = {
        "profile": profile.get("name"),
        "present": present,
        "missing_core": missing_core,
        "failed": failed,
        "pos_rates": rates,
        "guard": guard_notes,
        "n_rows": int(len(df)),
    }
    print("[headcheck] " + json.dumps(summary, ensure_ascii=False), flush=True)
    if missing_core:
        print(f"[headcheck] ВНИМАНИЕ: не посчитаны обязательные головы: "
              f"{missing_core}", flush=True)

    # дамп вероятностей для локального анализа (на сервере env не выставлен)
    dump_path = _os.environ.get("ECUP_DUMP_PROBS")
    if dump_path:
        pd.DataFrame({"id": df["id"].values, "prob_ok": probs}).to_csv(
            dump_path, index=False)
        print(f"[dump] вероятности -> {dump_path}", flush=True)
    budget.log("done")


def _emergency_fallback(argv) -> int:
    """Последний рубеж: что бы ни упало в main, записать валидный файл.

    Правило: сервер не показывает логи, битый файл = 0 баллов. Дефолт по
    категориям: БАД -> «не бан» (74% класса), ЛВЖ -> «бан» (96% класса).
    """
    try:
        from src.config import CAT_BAD
        args = parse_args()
        df = prepare_dataframe(Path(args.test_data_path))
        preds = (df["category"].values == CAT_BAD).astype(np.int64)
        write_submission(df, preds, template_comments(df, preds),
                         args.output_path)
        print("[fallback] записан аварийный файл с дефолтами категорий",
              flush=True)
        return 0
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(_emergency_fallback(sys.argv))
