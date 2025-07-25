import torch
import torch.nn as nn
from typing import List, Optional, Union

def build_ffnn_model(n_input_features: int,
                     n_output_features: int,
                     layer_widths: List[int],
                     use_layer_norm: bool,
                     apply_layer_norm_final_layer: bool = False,
                     act_fn_callable: nn.Module = nn.ReLU,
                     output_act_fn_callable: Optional[nn.Module] = None,
                     device: Union[str, torch.device] = "cpu") -> nn.Module:

    """
    Build a feedforward neural network model.
    
    :param n_input_features: Number of input features.
    :param n_output_features: Number of output features.
    :param layer_widths: List of widths for each hidden layer.

    :param use_layer_norm: Whether to use layer normalization. Note that this is applied between each linear layer and the corresponding activation function. 
                           It is NOT applied to the input tensors, nor is it applied after any activation functions. This is referred to as "pre-activation" normalization.

    :param apply_layer_norm_final_layer: Whether to apply layer normalization to the final layer. This will place a LayerNorm layer after the final linear layer. 
                                         If an output activation function is specified, it will be applied after the LayerNorm layer as with all other layers.
    
    :param act_fn_callable: Callable to instantiate activation function.
    :param output_act_fn_callable: Callable to instantiate output activation function.
    :param device: Device to put the model on.

    :return: A nn.Sequential module that implements the feedforward neural network.
    """
    
    layers = []

    in_widths = [n_input_features] + layer_widths
    out_widths = layer_widths + [n_output_features]

    for i in range(len(in_widths)):
        layers.append(nn.Linear(in_widths[i], out_widths[i]))
        if use_layer_norm:
            if i != len(in_widths) - 1 or (i == len(in_widths) - 1 and apply_layer_norm_final_layer):
                layers.append(nn.LayerNorm(out_widths[i]))
        if i != len(in_widths) - 1:
            layers.append(act_fn_callable())

    if output_act_fn_callable is not None:
        if output_act_fn_callable == torch.nn.Softmax:
            layers.append(output_act_fn_callable(dim=-1))
        else:
            layers.append(output_act_fn_callable())

    model = nn.Sequential(*layers).to(device)
    return model

