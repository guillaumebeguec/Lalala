"""PE-AV (Perception Encoder Audio-Visual, Meta) via transformers >= 5.21, on MPS.

Weights: `facebook/pe-av-large` (Apache 2.0, not gated according to the HF card). API
read from the model card and the processor signature:
    inputs = processor(audio=[np.ndarray...], text=[...], videos=[paths...], return_tensors="pt", padding=True)
    outputs = model(**inputs)
    outputs.audio_embeds / outputs.text_audio_embeds (text aligned to audio)
    outputs.video_embeds / outputs.text_video_embeds (text aligned to video)
The expected sampling rate is read from `processor.feature_extractor.sampling_rate`.
If the API differs at load time, we stop with an explicit error (no guessing).
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import torch

from core.audio_io import load_audio, resample, to_mono
from core.device import get_device
from core.embedders.base import AudioItem, Embedder

logger = logging.getLogger(__name__)

MODEL_ID = "facebook/pe-av-large"


class PEAVEmbedder(Embedder):
    name = "peav"
    dim = 0
    supports_video = True
    languages = "en"

    def __init__(self, model_id: str = MODEL_ID, text_space: str = "audio", autocast: bool = False):
        super().__init__()
        self.model_id = model_id
        self.text_space = text_space  # "audio": text aligned to audio; "video": aligned to video
        self.autocast = autocast      # fp16/bf16 on MPS: NaN or slower than fp32 (measured 28/10) → disabled
        self._model = None
        self._processor = None
        self.sr = None

    def version(self) -> str:
        return f"1-{self.text_space}"

    def _load(self):
        if self._model is not None:
            return
        from transformers import PeAudioVideoModel, PeAudioVideoProcessor
        self.device = get_device()
        logger.info("Loading PE-AV %s on %s", self.model_id, self.device)
        self._processor = PeAudioVideoProcessor.from_pretrained(self.model_id)
        # transformers 5.21 + timm 1.0.30: the vision config declares `num_classes=1024` but not
        # `num_labels`, and TimmWrapper builds a 2-class head → incompatible sizes when loading.
        # We align num_labels on the checkpoint before loading the weights.
        from transformers import PeAudioVideoConfig
        cfg = PeAudioVideoConfig.from_pretrained(self.model_id)
        vis = getattr(getattr(cfg.video_config, "video_config", None), "vision_config", None)
        if vis is not None and getattr(vis, "num_labels", None) in (None, 2) and getattr(vis, "num_classes", None):
            vis.num_labels = int(vis.num_classes)
        self._model = PeAudioVideoModel.from_pretrained(self.model_id, config=cfg, dtype=torch.float32).to(self.device).eval()
        fe = getattr(self._processor, "feature_extractor", None) or getattr(self._processor, "audio_processor", None)
        self.sr = int(getattr(fe, "sampling_rate", 0) or 0)
        if not self.sr:
            raise RuntimeError("PE-AV: audio sampling rate not found in the processor; check the API")
        logger.info("PE-AV: audio expected at %d Hz", self.sr)

    def _to_wave(self, item: AudioItem) -> np.ndarray:
        if isinstance(item, (bytes, bytearray)):
            x, _ = load_audio(bytes(item), sr=self.sr)
            return x
        x, sr = item
        return resample(to_mono(np.asarray(x)), sr, self.sr)

    def _inputs(self, **kw):
        # The full forward requires >= 2 modalities; for a single one, we go through the get_*_embeds methods.
        return self._processor(return_tensors="pt", padding=True, **kw).to(self.device)

    @staticmethod
    def _np(t: torch.Tensor) -> np.ndarray:
        return t.float().cpu().numpy()

    def _ctx(self):
        if self.autocast and self.device.type in ("mps", "cuda"):
            return torch.autocast(self.device.type, dtype=torch.float16)
        import contextlib
        return contextlib.nullcontext()

    def _release(self) -> None:
        """Release MPS memory between two batches: without this, the allocation grows until OOM (26 GB after ~900 previews)."""
        if self.device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()

    def _embed_audio(self, items: Sequence[AudioItem]) -> np.ndarray:
        self._load()
        waves = [self._to_wave(it) for it in items]
        try:
            inp = self._inputs(audio=waves, sampling_rate=self.sr)
            with torch.inference_mode(), self._ctx():
                emb = self._model.get_audio_embeds(inp["input_values"], padding_mask=inp.get("padding_mask"))
            out = self._np(emb)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower() or len(items) == 1:
                raise
            logger.warning("PE-AV: out of memory on a batch of %d, retrying in two halves", len(items))
            del waves
            self._release()
            h = len(items) // 2
            out = np.concatenate([self._embed_audio(items[:h]), self._embed_audio(items[h:])])
        finally:
            self._release()
        self.dim = out.shape[-1]
        return out

    def _embed_text(self, texts: Sequence[str], task: str) -> np.ndarray:
        self._load()
        inp = self._inputs(text=list(texts))
        # transformers 5.21: PeAudioVideoModel.get_text_audio_embeds calls a non-existent method,
        # and PeAudioModel.get_text_audio_embeds reads `hidden_states[-1]` without requesting them. We reproduce
        # the computation: last hidden state of the text model, [CLS] token, then the text→audio contrastive head.
        if self.text_space != "audio":
            raise NotImplementedError("PE-AV: only the text<->audio space is exposed (no text method on the video side)")
        sub = self._model.audio_model
        with torch.inference_mode(), self._ctx():
            out = sub.text_model(input_ids=inp["input_ids"], attention_mask=inp.get("attention_mask"), return_dict=True)
            emb = sub.text_audio_head(out.last_hidden_state[:, 0])
        self.dim = emb.shape[-1]
        res = self._np(emb)
        self._release()
        return res

    def _embed_video(self, paths: Sequence[str]) -> np.ndarray:
        self._load()
        inp = self._inputs(videos=list(paths))
        with torch.inference_mode(), self._ctx():
            emb = self._model.get_video_embeds(inp["pixel_values_videos"], padding_mask_videos=inp.get("padding_mask_videos"))
        return self._np(emb)
