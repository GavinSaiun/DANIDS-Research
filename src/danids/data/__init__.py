"""Leakage-safe dataset registry, manifests, loading, and preprocessing.

Public objects live in their focused submodules. Keeping this package initializer
free of eager imports also keeps configuration and manifest modules acyclic.
"""
