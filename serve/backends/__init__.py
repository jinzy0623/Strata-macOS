"""Inference plugins share Strata's token-id Engine protocol, tokenizer and template.

The legacy resident CUDA subprocess remains `--engine strata`. No CUDA module is
imported by this registry. Add a plugin with register_backend(name, factory).
"""
from dataclasses import dataclass
from typing import Callable


@dataclass
class BackendBundle:
    engine: object
    tokenizer: object
    template: object
    stop_ids: set[int]


_FACTORIES: dict[str, Callable[[dict], BackendBundle]] = {}


def register_backend(name: str, factory: Callable[[dict], BackendBundle]):
    if name in _FACTORIES:
        raise ValueError(f"backend already registered: {name}")
    _FACTORIES[name] = factory


def backend_names():
    return tuple(_FACTORIES)


def load_backend(name: str, config: dict) -> BackendBundle:
    try:
        factory = _FACTORIES[name]
    except KeyError:
        raise ValueError(f"unknown backend: {name}") from None
    return factory(config)


def _ggml(config, *, metal):
    from .ggml import create_backend
    return create_backend(config, metal=metal)


register_backend("metal", lambda config: _ggml(config, metal=True))
register_backend("ggml-cpu", lambda config: _ggml(config, metal=False))
