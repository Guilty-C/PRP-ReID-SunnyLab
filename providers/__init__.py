# providers/__init__.py
from __future__ import annotations
from typing import Dict, Type, Any
import importlib

# 注册表：provider_name -> Provider类
_REG: Dict[str, Type] = {}

def register(name: str):
    """用于在 provider 模块内装饰类，把类注册到全局表。"""
    def deco(cls):
        _REG[name] = cls
        return cls
    return deco

def _lazy_import_for(name: str) -> None:
    """
    按需导入 providers.<module>，以触发该模块中的 @register(...) 副作用。
    例如 name='clip-text-local' -> module 'clip_text_local'
    """
    mod = name.replace("-", "_")
    for cand in (mod, name):
        try:
            importlib.import_module(f"{__name__}.{cand}")
            return  # 成功导入即返回
        except ModuleNotFoundError:
            continue
        except Exception:
            # 其它异常不在这里抛，交给 get_provider 的 KeyError 统一报错
            return

def get_provider(name: str) -> Type:
    """返回已注册的 Provider类（注意：是类，不是实例）。"""
    if name not in _REG:
        _lazy_import_for(name)
    if name not in _REG:
        raise KeyError(f"Unknown provider: {name}. Available: {sorted(_REG.keys())}")
    return _REG[name]

def create_provider(name: str, **kwargs: Any):
    """
    返回 Provider 实例。优先调用类方法 from_env(**kwargs)，否则直接 cls(**kwargs)。
    """
    cls = get_provider(name)
    if hasattr(cls, "from_env") and callable(getattr(cls, "from_env")):
        return cls.from_env(**kwargs)
    return cls(**kwargs)
