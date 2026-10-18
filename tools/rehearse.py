"""СКВОЗНАЯ РЕПЕТИЦИЯ — обязательный этап перед сборкой архива.

  python tools/rehearse.py [--time-limit 14400]

Запускает ТОТ ЖЕ run.py тем же интерпретатором на локальном data/data.csv
(12 971 строка с известными ответами), затем:
  1. проверяет валидность выходного файла и полноту ответов;
  2. считает метрику соревнования и сверяет с полом;
  3. разбирает сводку [headcheck]: профиль обязан быть full, все
     обязательные головы — посчитаны, ни одного упавшего шага;
  4. пишет artifacts/manifest.json с хэшами ВСЕХ файлов, определяющих
     поведение сабмита. tools/make_submit.py без свежего манифеста
     архив не соберёт.

Урок, ради которого всё это: сабмиты №1 (0.71) и №2 (0.67) при домашней
оценке 0.92 упали на шагах, которые никто не прогонял тем же кодом дома.
"""
import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import norm_category  # noqa: E402
from src.manifest import collect_state, write_manifest  # noqa: E402
from src.validate import validate_submission  # noqa: E402
from train.metric import metric_a, report  # noqa: E402

DATA_CSV = ROOT / "data" / "data.csv"
OUT_CSV = ROOT / "data" / "work" / "rehearsal_submit.csv"
LOG_TXT = ROOT / "data" / "work" / "rehearsal_log.txt"
METRIC_FLOOR = 0.93          # репетиция на трейне оптимистична (дубли в базе)

_VERDICT_RE = re.compile(r"<вердикт>(бан|не бан)$")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-limit", type=int, default=14400,
                        help="ECUP_TIME_LIMIT_SEC для прогона (CPU медленнее H100)")
    args = parser.parse_args()

    if not DATA_CSV.exists():
        sys.exit(f"[rehearse] нет {DATA_CSV}")

    # ------------------------------------------------------- запуск run.py
    env = dict(os.environ)
    env["ECUP_TIME_LIMIT_SEC"] = str(args.time_limit)
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = [sys.executable, "-u", "run.py",
           "--test_data_path", str(DATA_CSV),
           "--output_path", str(OUT_CSV)]
    print(f"[rehearse] запуск: {' '.join(cmd)}", flush=True)
    t0 = time.monotonic()
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True,
                          text=True, encoding="utf-8", errors="replace")
    elapsed = time.monotonic() - t0
    LOG_TXT.write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr,
                       encoding="utf-8")
    print(f"[rehearse] завершён за {elapsed/60:.1f} мин, код {proc.returncode}, "
          f"лог: {LOG_TXT}", flush=True)
    if proc.returncode != 0:
        print(proc.stderr[-3000:])
        sys.exit("[rehearse] ПРОВАЛ: run.py завершился с ошибкой")

    # ------------------------------------------------------- сводка headcheck
    hc = None
    for line in proc.stdout.splitlines():
        if line.startswith("[headcheck] {"):
            hc = json.loads(line[len("[headcheck] "):])
    if hc is None:
        sys.exit("[rehearse] ПРОВАЛ: run.py не напечатал [headcheck]")
    problems = []
    if hc.get("profile") != "full":
        problems.append(f"профиль {hc.get('profile')}, ожидался full")
    if hc.get("missing_core"):
        problems.append(f"не посчитаны головы: {hc['missing_core']}")
    if hc.get("failed"):
        problems.append(f"упавшие шаги: {hc['failed']}")

    # ------------------------------------------------------- метрика
    truth = pd.read_csv(DATA_CSV)
    sub = pd.read_csv(OUT_CSV)
    errors = validate_submission(sub, expected_ids=truth["id"].tolist())
    if errors:
        sys.exit(f"[rehearse] ПРОВАЛ: файл невалиден: {errors[:5]}")

    verdicts = []
    for value in sub["result"].tolist():
        m = _VERDICT_RE.search(str(value))
        verdicts.append(1 if (m and m.group(1) == "не бан") else 0)
    merged = truth[["id", "label", "category"]].merge(
        pd.DataFrame({"id": sub["id"], "pred": verdicts}), on="id")
    y = merged["label"].values
    preds = merged["pred"].values.astype(np.int64)
    cats = merged["category"].map(norm_category).values
    a = metric_a(y, preds, cats)
    report(y, preds, cats, tag="rehearsal")
    if a < METRIC_FLOOR:
        problems.append(f"метрика репетиции {a:.4f} ниже пола {METRIC_FLOOR}")

    if problems:
        for p in problems:
            print(f"[rehearse] ПРОБЛЕМА: {p}", flush=True)
        sys.exit("[rehearse] ПРОВАЛ: манифест НЕ записан")

    # ------------------------------------------------------- манифест
    oof_a = None
    oof_report = ROOT / "data" / "work" / "oof_report.json"
    if oof_report.exists():
        oof_a = json.loads(oof_report.read_text(encoding="utf-8"))\
            .get("full", {}).get("oof_a")

    import platform
    import sklearn
    payload = {
        "rehearsed_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "rehearsal_metric_a": round(float(a), 4),
        "rehearsal_rows": int(len(merged)),
        "rehearsal_minutes": round(elapsed / 60, 1),
        "oof_metric_a": oof_a,
        "headcheck": hc,
        "env": {"python": platform.python_version(),
                "sklearn": sklearn.__version__},
        "files": collect_state(),
    }
    write_manifest(payload)
    print(f"[rehearse] УСПЕХ: метрика {a:.4f} (OOF {oof_a}), "
          f"манифест записан ({len(payload['files'])} файлов)", flush=True)


if __name__ == "__main__":
    main()
