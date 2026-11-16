def calculate_parameters(layers: list[dict]) -> int:
    """
    Calculate the total number of trainable parameters in a neural network.

    Args:
        layers: List of dictionaries, each describing a layer.

    Returns:
        Total number of trainable parameters (int).
    """
    # Your code here
    total=0
    for layer in layers:
        layer_type= layer["type"]
        bias=layer.get("bias",True)
        
        if layer_type=="dense":
            params=layer["input_size"]*layer["output_size"]

            if bias:
                params+=layer["output_size"]

        if layer_type=="conv2d":
            params=layer["in_channels"]*layer["out_channels"]*layer["kernel_size"]*layer["kernel_size"]

            if bias:
                params+=layer["out_channels"]

        total+=params
    return total
