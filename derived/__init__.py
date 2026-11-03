"""Derived parameters: their metadata and an emulator x -> (derived_1, ..., derived_k).

Every true evaluation records the backend's derived parameters (derived_data/). This
package trains a network on them, so that a surrogate chain can carry derived columns
and a likelihood on derived parameters can be evaluated during sampling.
"""
