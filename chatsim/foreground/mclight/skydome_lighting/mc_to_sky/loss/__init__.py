import importlib


def build_loss(loss_conf):
    loss_name = loss_conf['type']
    loss_args = loss_conf['args']

    loss_lib = importlib.import_module("mc_to_sky.loss.loss")
    loss_cls = None
    target_loss_name = loss_name.replace('_', '')

    for name, cls in loss_lib.__dict__.items():
        if name.lower() == target_loss_name.lower():
            loss_cls = cls

    return loss_cls(loss_args)