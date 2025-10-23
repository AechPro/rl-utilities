import torch
import torch.nn as nn
from typing import List, Optional, Union
from rl_utilities.torch_modules import GLU

def build_glu_model(n_input_features: int,
                     n_output_features: int,
                     layer_widths: List[int],
                     use_layer_norm: bool,
                     apply_layer_norm_final_layer: bool = False,
                     act_fn_callable: nn.Module = nn.Sigmoid,
                     output_act_fn_callable: Optional[nn.Module] = None,
                     device: Union[str, torch.device] = "cpu") -> nn.Module:

    """
    Build a gated linear unit model.
    
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

    :return: A nn.Sequential module that implements the gated linear unit.
    """
    
    layers = []

    in_widths = [n_input_features] + layer_widths
    out_widths = layer_widths + [n_output_features]

    for i in range(len(in_widths)):
        layers.append(GLU(in_widths[i], out_widths[i], act_fn_callable(), use_layer_norm))

    if apply_layer_norm_final_layer and use_layer_norm:
        layers.append(nn.LayerNorm(out_widths[-1]))
        
    if output_act_fn_callable is not None:
        if output_act_fn_callable == torch.nn.Softmax:
            layers.append(output_act_fn_callable(dim=-1))
        else:
            layers.append(output_act_fn_callable())

    model = nn.Sequential(*layers).to(device)
    return model

