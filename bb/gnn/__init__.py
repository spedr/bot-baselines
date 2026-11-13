"""Shared GNN skeleton for the botrgcn feature family (BotRGCN, GCN/GAT, RGT, Simple-HGN, ...).

    from bb.features import load_features
    from bb.gnn import FitConfig, fit_full_batch, kaiming_init_linears, make_classifier, split_metrics, to_device

    F = load_features(ds)                                         # memmaps (bb/features/botrgcn.py)
    J = prepare_extscore(args, ds, F.splits, labels=F.label, neighbours=...)
    g = to_device(F, "cuda", J=J, extscore_inject="num")              # ExtScore [z, m] appended to num_prop
    model = make_classifier(lambda: MyBody(D), hidden=D, num_dim=F.num_prop.shape[1] + J.width,
                            cat_dim=F.cat_prop.shape[1], extscore_width=J.width, extscore_inject="num").to("cuda")
    kaiming_init_linears(model)
    res = fit_full_batch(model, g, F.splits, FitConfig(epochs=200, lr=1e-2, weight_decay=5e-2), run=run)
    metrics = split_metrics(res.prob, F.label, F.splits)

See docs/architecture.md section 12.
"""
from .inputs import EXTSCORE_INJECT, GraphInputs, to_device
from .model import GNNClassifier, InputEncoder, OutputHead, RGCNConv, kaiming_init_linears, make_classifier, n_params
from .train import FitConfig, FitResult, fit_full_batch, fit_sampled, predict_full, predict_sampled, split_metrics

__all__ = ["EXTSCORE_INJECT", "GraphInputs", "to_device", "GNNClassifier", "InputEncoder", "OutputHead", "RGCNConv",
           "kaiming_init_linears", "make_classifier", "n_params", "FitConfig", "FitResult", "fit_full_batch",
           "fit_sampled", "predict_full", "predict_sampled", "split_metrics"]
