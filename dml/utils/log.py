"""
Journalisation : messages à la console (donc dans le .out SLURM) et dans <run_dir>/log.txt.

    from logging_utils import setup_logging
    log = setup_logging()              # au début du script
    setup_logging(run_dir)             # une fois le dossier de résultats connu : ajoute log.txt
    log.info("message")

Dans les autres modules : log = logging.getLogger(__name__).
"""
import logging
import os
import sys
from typing import Optional

_FORMAT = "%(asctime)s %(levelname)-7s %(message)s"
_DATEFMT = "%H:%M:%S"


def setup_logging(run_dir: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Configure le logger racine (une seule fois) et ajoute un fichier log.txt si run_dir est donné."""
    root = logging.getLogger()
    root.setLevel(level)
    if not any(getattr(h, "_dml_console", False) for h in root.handlers):
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter(_FORMAT, _DATEFMT))
        console._dml_console = True
        root.addHandler(console)
    if run_dir is not None:
        path = os.path.abspath(os.path.join(run_dir, "log.txt"))
        if not any(getattr(h, "baseFilename", None) == path for h in root.handlers):
            os.makedirs(run_dir, exist_ok=True)
            handler = logging.FileHandler(path, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
            root.addHandler(handler)
    # TensorFlow et absl sont très bavards au niveau INFO
    logging.getLogger("tensorflow").setLevel(logging.WARNING)
    logging.getLogger("absl").setLevel(logging.WARNING)
    return logging.getLogger("dml")
