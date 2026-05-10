#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Uso:
  ./scripts/setup_env.sh [--fresh] [PROJECT_PATH]

Opciones:
  --fresh       Borra .venv antes de instalar
  -h, --help    Muestra esta ayuda

Ejemplos:
  ./scripts/setup_env.sh
  ./scripts/setup_env.sh --fresh
  ./scripts/setup_env.sh /ruta/proyecto
EOF
}

FRESH=false
PROJECT_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fresh)
      FRESH=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      if [[ -n "$PROJECT_PATH" ]]; then
        echo "Error: solo se permite una ruta de proyecto." >&2
        usage
        exit 1
      fi
      PROJECT_PATH="$1"
      shift
      ;;
  esac
done

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv no está instalado o no está en PATH." >&2
  echo "Instalación: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

if [[ -z "$PROJECT_PATH" ]]; then
  PROJECT_PATH="$(pwd)"
fi

PROJECT_PATH="$(cd "$PROJECT_PATH" && pwd)"

if [[ ! -f "$PROJECT_PATH/pyproject.toml" ]]; then
  echo "Error: no se encontró pyproject.toml en $PROJECT_PATH" >&2
  exit 1
fi

echo "==> Proyecto: $PROJECT_PATH"

if [[ "$FRESH" == "true" ]]; then
  echo "==> Eliminando entorno previo: $PROJECT_PATH/.venv"
  rm -rf "$PROJECT_PATH/.venv"
fi

echo "==> Instalando dependencias con uv (Python 3.12)"
(
  cd "$PROJECT_PATH"
  uv sync --python 3.12
)

echo "==> Validando paquetes y CUDA"
(
  cd "$PROJECT_PATH"
  uv run python - <<'PY'
import importlib

packages = ["torch", "transformers", "huggingface_hub"]
for name in packages:
    module = importlib.import_module(name)
    version = getattr(module, "__version__", "desconocida")
    print(f"[OK] {name}: {version}")

import torch
print(f"CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
else:
    print("GPU detectada: ninguna")
PY
)

echo "==> Entorno listo"
