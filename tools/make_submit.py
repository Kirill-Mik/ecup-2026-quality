"""Сборка архива сабмита. БЕЗ свежей репетиции архив не собирается.

  python tools/make_submit.py [--out ecup2_submit3.zip]

Гейты (все обязательны, обхода нет):
  1. artifacts/manifest.json существует — репетиция проводилась;
  2. хэши ВСЕХ файлов совпадают с манифестом — с момента репетиции ничего
     не менялось (иначе на сервер поедет непроверенное);
  3. метрика репетиции не ниже пола;
  4. в src/ нет запрещённых модулей (bert/judge/ocr/vl — их нельзя
     отрепетировать дома, значит им нельзя на сервер).

В архив кладётся РОВНО состав манифеста + сам манифест: ни одного файла,
не прошедшего репетицию.
"""
import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.manifest import MANIFEST_FILE, load_manifest, verify_manifest  # noqa: E402

MAX_ARCHIVE_BYTES = 5 * 1024 ** 3
METRIC_FLOOR = 0.93
FORBIDDEN_MODULES = {"bert_head.py", "embed.py", "gbm.py", "imghash.py",
                     "judge.py", "llm_cache.py", "lora.py", "ocr.py",
                     "vl_cls.py", "vlknn_vl.py"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="ecup2_submit4.zip")
    args = parser.parse_args()

    # ------------------------------------------------------------- гейты
    man = load_manifest()
    if not man:
        sys.exit("[submit] ОТКАЗ: нет artifacts/manifest.json — сначала "
                 "python tools/rehearse.py")
    a = man.get("rehearsal_metric_a") or 0.0
    if a < METRIC_FLOOR:
        sys.exit(f"[submit] ОТКАЗ: метрика репетиции {a} ниже пола {METRIC_FLOOR}")

    print("[submit] сверяю хэши с манифестом (около минуты)...", flush=True)
    problems = verify_manifest(check_hashes=True)
    if problems:
        for p in problems[:20]:
            print(f"[submit] ПРОБЛЕМА: {p}")
        sys.exit("[submit] ОТКАЗ: состав изменился после репетиции — "
                 "повторите python tools/rehearse.py")

    for rel in man["files"]:
        name = rel.split("/")[-1]
        if name in FORBIDDEN_MODULES:
            sys.exit(f"[submit] ОТКАЗ: запрещённый модуль в составе: {rel}")

    # ------------------------------------------------------------- сборка
    out_path = ROOT / args.out
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in sorted(man["files"]):
            zf.write(ROOT / rel, rel)
        zf.write(MANIFEST_FILE, "artifacts/manifest.json")

    size = out_path.stat().st_size
    print(f"[submit] {out_path.name}: {size / 1024 ** 2:.1f} МБ, "
          f"{len(man['files']) + 1} файлов")
    if size > MAX_ARCHIVE_BYTES:
        out_path.unlink()
        sys.exit("[submit] ОТКАЗ: превышен лимит архива 5 ГБ")

    print(f"[submit] ГОТОВО. Репетиция: A={a} за {man.get('rehearsal_minutes')} "
          f"мин ({man.get('rehearsed_at')}), OOF A={man.get('oof_metric_a')}")
    print("[submit] профиль репетиции:",
          man.get("headcheck", {}).get("profile"),
          "| доли ok:", man.get("headcheck", {}).get("pos_rates"))


if __name__ == "__main__":
    main()
