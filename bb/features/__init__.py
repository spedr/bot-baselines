"""Feature schemes shared by several methods (built once per dataset, read by many).

- ``botrgcn``: description embedding, tweet-mean embedding, numeric and categorical property
  blocks, follow graph, labels and splits (docs/methods/botrgcn.md 3.1). Consumers: BotRGCN,
  GCN/GAT, RGT, Simple-HGN/HGT, BotMoE, BotDGT, SEBot, BIC, SEGCN.

    from bb.features import load_features
    F = load_features("twibot22")        # BotRGCNFeatures (memmaps); builds if missing or stale
"""
_EXPORTS = ("BotRGCNFeatures", "VARIANTS", "build_features", "ensure_features", "feature_dir", "load_features")
__all__ = list(_EXPORTS)


def __getattr__(name):  # lazy, so `python -m bb.features.botrgcn` does not import the module twice
    if name in _EXPORTS:
        from . import botrgcn
        return getattr(botrgcn, name)
    raise AttributeError(name)
