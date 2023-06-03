from collections import defaultdict
import torch

def get_mup_multipliers(base_model, main_model):
    """
    Make a dict of name:multiplier for each parameter in main model
    """
    base_shapes = _get_shapes(base_model)
    model_shapes = _get_shapes(main_model)
    basenames = set(base_shapes.keys())
    names = set(model_shapes.keys())
    assert basenames == names, (
        f"`base_shapes` has extra names {basenames - names}. " f"`shapes` has extra names {names - basenames}."
    )
    multipliers = {}
    for name, b_shape in base_shapes.items():
        multipliers[name] = _get_multiplier(b_shape, model_shapes[name])
    return multipliers


def _get_multiplier(base_dims, dims):
    # the 'multiplier' is the ratio of dim / base_dim for the **last dimension** that is infinite
    # the weight is 'matrix like' if it has >1 infinite dimension
    # eg if base_dims=[d1, d2_base] and dims=[d1, d2] we would return (d2/d2_base, False)
    num_inf_dims = 0
    multiplier = 1
    for base_dim, dim in zip(base_dims, dims):
        assert isinstance(base_dim, int), f"Unknown base_dim type: {type(base_dim)}"
        if base_dim != dim:
            num_inf_dims += 1
            multiplier = dim / base_dim
    is_matrix_like = True if num_inf_dims > 1 else False
    return (multiplier, is_matrix_like)


def mup_init(model, mup_multipliers_dict):
    for name, param in model.named_parameters():
        if "layers" in name:
            name = ".".join(name.split(".")[2:])
        if "bias" in name and name in mup_multipliers_dict:
            param.data *= mup_multipliers_dict[name][0] ** 0.5


def build_optimizer_param_groups(model, mup_multipliers_dict, decoupled_wd=False, **optimizer_kwargs):
    """
    MuP scales the lr according to if a param is 'matrix like' or 'vector like'
    We build params_groups based on this scaled lr
    """

    def new_group():
        new_g = {k: v for k, v in optimizer_kwargs.items()}
        new_g["params"] = []
        return new_g

    param_groups = defaultdict(new_group)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "_fsdp_wrapped_module." in name:
            name = name.split("_fsdp_wrapped_module.")[-1]
        else:
            if name.startswith("module."):
                name = name.split("module.")[-1]
            if "layers" in name:
                name = ".".join(name.split(".")[2:])
        multiplier, is_matrix_like = mup_multipliers_dict[name]
        if is_matrix_like:
            param_groups[multiplier]["params"].append(param)
        else:
            param_groups[1.0]["params"].append(param)

    for width_mult, group in param_groups.items():
        # Scale learning rate and weight decay accordingly
        group["lr"] /= width_mult
        if not decoupled_wd:
            group["weight_decay"] *= width_mult

    return list(param_groups.values())


def _get_shapes(model):
    """
    Returns a dictioanry of name:shape for each unique layer in a model.
    If a model comprises multiple 'blocks' (eg TransformerBlocks)
    we assume every block has the same dimensions
    """
    shapes_dict = {}
    for name, param in model.named_parameters():
        if "layers.0" in name:
            name = ".".join(name.split(".")[2:])
            shapes_dict[name] = param.shape
        elif "layers" in name:
            name = ".".join(name.split(".")[2:])
            assert shapes_dict[name] == param.shape, "_get_shapes assumes all blocks have the same dimensions"
        else:
            shapes_dict[name] = param.shape
    return shapes_dict


class MuReadout(torch.nn.Linear):
    """Drop-in replacement for all output linear layers.

    An "output" linear layer is one that maps from a width dimension (e.g.,
    `d_model` in a Transformer) to a non-width dimension (e.g., vocab size).

    This layer implements the version of μP with a 1/width multiplier and a
    constant variance initialization for both weights and biases.
    """

    def __init__(self, width_mult, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        self.width_mult = width_mult
        self._has_rescaled_params = False
        self._rescale_parameters()

    def _rescale_parameters(self):
        """
        Rescale parameters to convert SP initialization to μP initialization.
        Warning: This method is NOT idempotent and should be called only once
        unless you know what you are doing.
        """
        if self._has_rescaled_params:
            raise RuntimeError("`_rescale_parameters` has been called once before already.")
        if self.bias is not None:
            self.bias.data *= self.width_mult ** 0.5
        self.weight.data *= self.width_mult ** 0.5
        self._has_rescaled_params = True

    def forward(self, x):
        return super().forward(x / self.width_mult)
