"""Paquet `core`. Réglages d'environnement à faire avant tout import de torch/faiss.

Sur macOS, faiss-cpu et PyTorch embarquent chacun un runtime OpenMP (libomp). Le second
chargement avorte le processus (« OMP: Error #15 ») ; KMP_DUPLICATE_LIB_OK=TRUE le laisse
continuer, mais des plantages (SIGSEGV dans __kmp_suspend) surviennent dès qu'un worker
OpenMP est lancé (observé le 28/10 avec CoverHunter). OMP_NUM_THREADS=1 supprime les
workers : les calculs lourds sont sur MPS, et FAISS sur 25 k vecteurs reste instantané.
Ces valeurs sont des défauts : l'environnement peut les surcharger."""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
