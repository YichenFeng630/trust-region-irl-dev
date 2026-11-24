"""Per-dimension linear normalisation to [-1, 1] ("limits" mode), as the official LinearNormalizer
(diffusion_policy/model/common/normalizer.py, _fit with mode='limits', output range [-1, 1], range_eps 1e-4,
fit_offset True). Dimensions whose range is below range_eps are only shifted, not scaled."""
import numpy as np
import jax.numpy as jnp


def fit_limits(data, output_min=-1.0, output_max=1.0, range_eps=1e-4):
    x = np.asarray(data, dtype=np.float32).reshape(-1, np.shape(data)[-1])
    input_min, input_max = x.min(axis=0), x.max(axis=0)
    input_range = input_max - input_min
    ignore_dim = input_range < range_eps
    input_range[ignore_dim] = output_max - output_min
    scale = (output_max - output_min) / input_range
    offset = output_min - scale * input_min
    offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]
    return {"scale": scale.astype(np.float32), "offset": offset.astype(np.float32)}


def normalize(params, x):
    return x * params["scale"] + params["offset"]


def unnormalize(params, x):
    return (x - params["offset"]) / params["scale"]


def to_jax(params):
    return {k: jnp.asarray(v) for k, v in params.items()}
