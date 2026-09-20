"""Manifest-driven CSGO Benchmark v2 Seen-10 support for Show-o2."""

from .data import CSGOSeen10Dataset, MAPS
from .model import CSGOSeen10Model, RadarPoseFiLM

__all__ = ["CSGOSeen10Dataset", "MAPS", "CSGOSeen10Model", "RadarPoseFiLM"]
