"""PyTorch device selection: MPS (Apple Silicon) first, CPU as fallback.

All code that places a tensor or a model on a device goes through here,
so that the fallback is consistent across the whole project.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_device() -> "torch.device":  # noqa: F821 — deferred import to keep the module lightweight
    """Return the device to use: `mps` if available, otherwise `cpu`.

    The `LALALA_DEVICE` environment variable forces a device (useful for
    tests or to compare MPS and CPU).
    """
    import torch

    forced = os.environ.get("LALALA_DEVICE")
    if forced:
        logger.info("Device forced by LALALA_DEVICE=%s", forced)
        return torch.device(forced)

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        logger.info("Device: mps (Apple Silicon)")
        return torch.device("mps")

    logger.warning("MPS unavailable, falling back to CPU")
    return torch.device("cpu")


def device_name() -> str:
    """Short device name, for logs and reports."""
    return str(get_device())
