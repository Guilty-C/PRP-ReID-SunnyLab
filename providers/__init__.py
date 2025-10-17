
_REG = {}

def register(name_or_cls, cls=None):
    """
    用法1：register("name", Class)
    用法2：@register("name") 置于类定义上方
    """
    if cls is None:
        name = name_or_cls
        def deco(c):
            _REG[name] = c
            return c
        return deco
    else:
        _REG[name_or_cls] = cls

def get_provider(name: str):
    if name not in _REG:
        raise KeyError(f"Unknown provider: {name}. Available: {sorted(_REG.keys())}")
    return _REG[name]

def create_provider(name: str, **kwargs):
    cls = get_provider(name)
    return cls.from_env(**kwargs) if hasattr(cls, "from_env") else cls(**kwargs)

# 触发 clip_text_local 导入，从而执行其 @register 装饰器
try:
    from . import clip_text_local as _clip_text_local  # noqa: F401
except Exception as _e:
    # 保持模块可导入，后续再报错
    pass
