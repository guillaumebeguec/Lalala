#!/bin/zsh
# SAM Audio install in a separate venv (perception-models requires decord, no macOS arm64 wheel → eva-decord + --no-deps)
R="$(cd "$(dirname "$0")/.." && pwd)"
PY=$R/data/raw/sam_audio/venv/bin/python
set -x
uv pip install --python $PY "setuptools<81" || exit 1
uv pip install --python $PY torch torchaudio torchvision torchcodec torchdiffeq einops numpy pydub "transformers>=4.54,<5" "huggingface_hub>=0.30,<1" audiobox_aesthetics eva-decord || exit 1
uv pip install --python $PY --no-deps "perception-models@git+https://github.com/facebookresearch/perception_models@unpin-deps" || exit 1
uv pip install --python $PY "dacvae@git+https://github.com/facebookresearch/dacvae.git" "imagebind@git+https://github.com/facebookresearch/ImageBind.git" "laion-clap@git+https://github.com/lematt1991/CLAP.git" || exit 1
uv pip install --python $PY --no-deps $R/data/raw/sam_audio/repo || exit 1
$PY -c "import sam_audio; print('IMPORT OK', sam_audio.__file__)"
# xformers has no macOS wheel: install a stub exposing the names imported at module load time
# (attention goes through PyTorch SDPA on MPS/CPU, the stub functions are never called)
SP=$($PY -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
mkdir -p $SP/xformers
cat > $SP/xformers/__init__.py <<'PY'
"""Stub for xformers on macOS: only the names imported by perception_models at load time."""
from . import ops, profiler  # noqa: F401
PY
cat > $SP/xformers/ops.py <<'PY'
"""Stub: attention goes through PyTorch; these are never called on MPS/CPU."""
import types


class AttentionBias:  # isinstance() target only
    pass


def _unavailable(*a, **k):
    raise RuntimeError("xformers is not available on macOS (stub)")


class _AttnBias:
    class LowerTriangularFromBottomRightMask(AttentionBias):
        pass

    class BlockDiagonalMask(AttentionBias):
        from_seqlens = staticmethod(_unavailable)

    class BlockDiagonalCausalMask(AttentionBias):
        from_seqlens = staticmethod(_unavailable)


fmha = types.SimpleNamespace(
    memory_efficient_attention=_unavailable,
    attn_bias=_AttnBias,
    flash=types.SimpleNamespace(FwOp=types.SimpleNamespace(OPERATOR=None)),
)
memory_efficient_attention = _unavailable
PY
cat > $SP/xformers/profiler.py <<'PY'
"""Stub: profiling is never used here."""
import contextlib


class MemSnapshotsProfiler:
    pass


class PyTorchProfiler:
    pass


@contextlib.contextmanager
def profile(*a, **k):
    yield
PY
echo "SAM INSTALL FIN"
