def find_first_conv_layer(model, layer_type, in_channels):
    for _, module in model.named_modules():
        if isinstance(module, layer_type) and \
                module.in_channels == in_channels:
            return module
