#!/bin/zsh
# Installation SAM Audio dans un venv séparé (perception-models exige decord, sans roue macOS arm64 → eva-decord + --no-deps)
R="$(cd "$(dirname "$0")/.." && pwd)"
PY=$R/data/raw/sam_audio/venv/bin/python
set -x
uv pip install --python $PY "setuptools<81" || exit 1
uv pip install --python $PY torch torchaudio torchvision torchcodec torchdiffeq einops numpy pydub "transformers>=4.54,<5" "huggingface_hub>=0.30,<1" audiobox_aesthetics eva-decord || exit 1
uv pip install --python $PY --no-deps "perception-models@git+https://github.com/facebookresearch/perception_models@unpin-deps" || exit 1
uv pip install --python $PY "dacvae@git+https://github.com/facebookresearch/dacvae.git" "imagebind@git+https://github.com/facebookresearch/ImageBind.git" "laion-clap@git+https://github.com/lematt1991/CLAP.git" || exit 1
uv pip install --python $PY --no-deps $R/data/raw/sam_audio/repo || exit 1
$PY -c "import sam_audio; print('IMPORT OK', sam_audio.__file__)"
echo "SAM INSTALL FIN"
