"""Paquet `core`. Réglage d'environnement à faire avant tout import de torch/faiss :
sur macOS, faiss-cpu et PyTorch embarquent chacun un runtime OpenMP (libomp) et le second
chargement avorte le processus (« OMP: Error #15 »). KMP_DUPLICATE_LIB_OK=TRUE est le
contournement documenté par Intel/LLVM ; on le pose ici car `core` est importé en premier."""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
