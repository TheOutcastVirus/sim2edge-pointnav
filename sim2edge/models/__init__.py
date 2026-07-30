"""Neural networks whose inference paths are shared by training and deploy."""

from .nav_policy import NavPolicy, NavPolicyConfig

__all__ = ["NavPolicy", "NavPolicyConfig"]
