from mc_to_sky.model.sub_module.basic_modules import (EncoderNet, 
                                                       DecoderNet, 
                                                       UNet, 
                                                       MLP)

def build_module(args):
    module_type = args['type']
    module_args = args['args']

    module_cls = eval(module_type)
    return module_cls(module_args)