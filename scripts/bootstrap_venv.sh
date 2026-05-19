#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
GET_PIP="${ROOT_DIR}/.cache/get-pip.py"

mkdir -p "${ROOT_DIR}/.cache"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  rm -rf "${VENV_DIR}"
  python3 -m venv --without-pip "${VENV_DIR}"
fi

if [[ ! -f "${GET_PIP}" ]]; then
  curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "${GET_PIP}"
fi

"${VENV_DIR}/bin/python" "${GET_PIP}"
"${VENV_DIR}/bin/pip" install -e "${ROOT_DIR}"

echo "Virtual environment ready at: ${VENV_DIR}"
echo "Activate with: source ${VENV_DIR}/bin/activate"
