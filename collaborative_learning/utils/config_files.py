"""
Lecture des fichiers de config (YAML ou JSON) et composition avec `defaults:`.

Un fichier YAML peut reprendre d'autres fichiers du dossier configs/ avant ses propres valeurs :

    # configs/methods/dml.yaml
    defaults: [defaults]          # configs/defaults.yaml d'abord
    loss_type: dml
    loss_form: additive

Les fichiers de `defaults` sont lus dans l'ordre (chacun peut avoir ses propres defaults), puis les
clés du fichier lui-même : la dernière valeur lue gagne. Un nom sans extension cherche .yaml,
.yml puis .json dans configs/ ; un chemin existant est pris tel quel.

Les fichiers JSON (anciennes configs, config.json enregistré dans un dossier de résultats)
restent lisibles.
"""
import json
import os
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_DIR = os.path.join(ROOT, "configs")
EXTENSIONS = (".yaml", ".yml", ".json")


def _load_yaml(path: str) -> Any:
    try:
        import yaml
    except ImportError as e:   # module PyYAML absent (cluster : ml PyYAML/...)
        raise SystemExit(f"Lecture de {path} impossible : PyYAML n'est pas installé "
                         "(pip install pyyaml, ou sur le cluster : ml PyYAML).") from e
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_config_path(name: str, base_dir: Optional[str] = None) -> str:
    """Chemin d'un fichier de config : chemin existant, sinon configs/<name>(.yaml|.yml|.json)."""
    if os.path.isfile(name):
        return name
    for root in (base_dir, CONFIG_DIR):
        if root is None:
            continue
        for candidate in [os.path.join(root, name)] + [os.path.join(root, name + ext) for ext in EXTENSIONS]:
            if os.path.isfile(candidate):
                return candidate
    raise SystemExit(f"Fichier de config introuvable : {name} (cherché tel quel et dans {CONFIG_DIR})")


def read_raw(path: str) -> Dict[str, Any]:
    """Contenu d'un fichier, sans composition."""
    if path.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = _load_yaml(path)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise SystemExit(f"{path} : un fichier de config doit contenir des paires clé: valeur")
    return data


def read_config_file(name: str, _seen: Optional[List[str]] = None) -> Dict[str, Any]:
    """Valeurs d'un fichier de config, après ses `defaults` (récursif)."""
    path = os.path.abspath(resolve_config_path(name))
    seen = list(_seen or [])
    if path in seen:
        raise SystemExit("Boucle dans les defaults : " + " -> ".join(seen + [path]))
    data = read_raw(path)
    defaults = data.pop("defaults", None) or []
    if isinstance(defaults, str):
        defaults = [defaults]
    values: Dict[str, Any] = {}
    for parent in defaults:
        values.update(read_config_file(parent, seen + [path]))
    values.update(data)
    return values


def write_yaml(path: str, values: Dict[str, Any], header: str = "") -> None:
    """Écrit un fichier YAML lisible (ordre des clés gardé), de façon atomique."""
    import yaml
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        if header:
            f.write("".join(f"# {line}\n" for line in header.splitlines()))
        yaml.safe_dump(values, f, sort_keys=False, allow_unicode=True, default_flow_style=None)
    os.replace(tmp, path)
