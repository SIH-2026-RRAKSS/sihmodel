"""
Adapters Package ? Multi-Dataset Loaders for AML and Mule-Chain Detection
=========================================================================
Provides isolated, honest data adapters for:
1. Synthetic Domestic Typology Dataset (Dataset A)
2. IBM AML Bank Transaction Graph (Dataset B)
3. Elliptic Bitcoin Illicit Flow Graph (Dataset C)
"""

from src.adapters.synthetic_adapter import SyntheticAdapter
from src.adapters.ibm_adapter import IBMAMLAdapter
from src.adapters.elliptic_adapter import EllipticAdapter

__all__ = ["SyntheticAdapter", "IBMAMLAdapter", "EllipticAdapter"]
