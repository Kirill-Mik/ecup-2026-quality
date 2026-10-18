"""Манифест сабмита: что именно проверено репетицией и какими файлами.

Манифест пишет ТОЛЬКО tools/rehearse.py — после успешной сквозной репетиции
тем же кодом и теми же артефактами, что едут на сервер. tools/make_submit.py
отказывается собирать архив, если манифест отсутствует или файлы изменились
после репетиции. run.py на сервере сверяет состав и громко печатает
расхождения (изменить результат он уже не может — но печать останется в
логах организаторов и в репетиции).
"""
import hashlib
import json
from pathlib import Path
from typing import Dict, List

from src.config import ARTIFACTS_DIR, ROOT

MANIFEST_FILE = ARTIFACTS_DIR / "manifest.json"
_BIG_FILE_BYTES = 64 * 1024 * 1024      # крупные файлы хэшируем частично


def file_digest(path: Path) -> str:
    """sha256 файла; для файлов >64МБ — sha256(голова 8МБ + хвост 8МБ + размер).

    Частичный хэш в 8000 раз быстрее на весах модели и всё равно ловит
    подмену/обрыв файла — единственные реальные режимы порчи артефакта.
    """
    size = path.stat().st_size
    h = hashlib.sha256()
    with open(path, "rb") as f:
        if size <= _BIG_FILE_BYTES:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        else:
            h.update(f.read(8 << 20))
            f.seek(-(8 << 20), 2)
            h.update(f.read(8 << 20))
            h.update(str(size).encode())
    return h.hexdigest()


def tracked_files() -> List[Path]:
    """Всё, что определяет поведение сабмита: код + артефакты."""
    files = [ROOT / "run.py", ROOT / "metadata.json"]
    files += sorted((ROOT / "src").glob("*.py"))
    files += sorted(p for p in ARTIFACTS_DIR.rglob("*")
                    if p.is_file() and p != MANIFEST_FILE)
    return files


def collect_state() -> Dict[str, str]:
    return {str(p.relative_to(ROOT)).replace("\\", "/"): file_digest(p)
            for p in tracked_files()}


def load_manifest() -> dict:
    if not MANIFEST_FILE.exists():
        return {}
    try:
        return json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def write_manifest(payload: dict) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def verify_manifest(check_hashes: bool = True) -> List[str]:
    """Список проблем; пусто = состав совпадает с отрепетированным."""
    man = load_manifest()
    if not man:
        return ["манифест отсутствует — репетиция не проводилась"]
    problems = []
    recorded = man.get("files", {})
    current = {str(p.relative_to(ROOT)).replace("\\", "/"): p
               for p in tracked_files()}
    for rel in recorded:
        if rel not in current:
            problems.append(f"файл из манифеста пропал: {rel}")
    for rel in current:
        if rel not in recorded:
            problems.append(f"файл не проходил репетицию: {rel}")
    if check_hashes:
        for rel, digest in recorded.items():
            p = current.get(rel)
            if p is not None and file_digest(p) != digest:
                problems.append(f"файл изменён после репетиции: {rel}")
    return problems
