"""CPU parity of the Flax Diffusion Policy port against the official PyTorch code, module by module.

Needs a checkout of the official code (real-stanford/diffusion_policy, read at 5ba07ac) and CPU torch, diffusers,
einops, numba and zarr<3. The checkout is found at $DIFFUSION_POLICY_REF, else at ../refs/diffusion_policy next to
this repository; without it (or without torch / diffusers) the module is skipped.

    JAX_PLATFORMS=cpu python -m pytest tests/test_dp_parity.py -q

Every comparison uses the same inputs, weights and random draws on both sides; the maximum absolute differences are
written to results/dp_parity_cpu.json.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("einops")
pytest.importorskip("numba")
pytest.importorskip("zarr")

REPO = Path(__file__).resolve().parents[1]
REF = Path(os.environ.get("DIFFUSION_POLICY_REF", REPO.parent / "refs" / "diffusion_policy"))
if not (REF / "diffusion_policy" / "model" / "diffusion" / "conditional_unet1d.py").exists():
    pytest.skip(f"official diffusion_policy checkout not found at {REF}", allow_module_level=True)
sys.path.insert(0, str(REF))

import jax                                                          # noqa: E402
import jax.numpy as jnp                                             # noqa: E402
import optax                                                        # noqa: E402
import diffusers                                                    # noqa: E402
import diffusers.schedulers.scheduling_ddpm as diffusers_ddpm       # noqa: E402
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler      # noqa: E402

from diffusion_policy.model.diffusion import conditional_unet1d as ref_unet       # noqa: E402
from diffusion_policy.model.diffusion import conv1d_components as ref_conv        # noqa: E402
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb as RefPosEmb   # noqa: E402
from diffusion_policy.model.diffusion.ema_model import EMAModel                   # noqa: E402
from diffusion_policy.model.common.normalizer import LinearNormalizer             # noqa: E402
# The official lr_scheduler.get_scheduler only dispatches "cosine" to this diffusers function; the wrapper itself
# imports names (Union, Optional, Optimizer) that newer diffusers no longer re-export, so it is called directly.
from diffusers.optimization import get_cosine_schedule_with_warmup               # noqa: E402
from diffusion_policy.common.sampler import SequenceSampler                       # noqa: E402
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy   # noqa: E402

from trust_region_irl.algorithms.diffusion_policy.flax_full_jit import unet, scheduler, policy as dp   # noqa: E402
from trust_region_irl.algorithms.diffusion_policy.flax_full_jit.normalizer import fit_limits, normalize, unnormalize   # noqa: E402
from trust_region_irl.algorithms.diffusion_policy.flax_full_jit.dataset import create_indices, gather_index   # noqa: E402

RECORDS = {}
CPU = jax.devices("cpu")[0]


@pytest.fixture(autouse=True, scope="module")
def _cpu_and_record():
    torch.manual_seed(0)
    with jax.default_device(CPU):
        yield
    out = REPO / "results" / "dp_parity_cpu.json"
    out.parent.mkdir(exist_ok=True)
    meta = {"torch": torch.__version__, "diffusers": diffusers.__version__, "jax": jax.__version__,
            "reference": str(REF), "device": "cpu"}
    out.write_text(json.dumps({"meta": meta, "results": RECORDS}, indent=1))


def record(name, ours, ref, tol, note=""):
    ours, ref = np.asarray(ours, np.float64), np.asarray(ref, np.float64)
    assert ours.shape == ref.shape, (name, ours.shape, ref.shape)
    diff = float(np.abs(ours - ref).max()) if ours.size else 0.0
    RECORDS[name] = {"max_abs_diff": diff, "max_abs_ref": float(np.abs(ref).max()) if ref.size else 0.0,
                     "tolerance": tol, "shape": list(ref.shape), "note": note}
    assert diff <= tol, f"{name}: max |diff| {diff:.3e} > {tol:.1e}"


# ---------------------------------------------------------------- weights: PyTorch state_dict -> Flax params #
def torch_to_flax(state_dict):
    """Map an official state_dict onto this port's parameter tree (same names, shapes and layouts)."""
    params = {}
    for key, value in state_dict.items():
        *path, leaf = key.split(".")
        if path[0] == "diffusion_step_encoder":
            out = [f"diffusion_step_encoder_{path[1]}"]
            rest = []
        elif path[0] in ("down_modules", "up_modules"):
            out, rest = [f"{path[0]}_{path[1]}_{path[2]}"], path[3:]
        elif path[0] in ("mid_modules", "final_conv"):
            out, rest = [f"{path[0]}_{path[1]}"], path[2:]
        else:                      # a single block tested on its own
            out, rest = [], path
        j = 0
        while j < len(rest):
            if rest[j] == "blocks":
                out.append(f"blocks_{rest[j + 1]}"); j += 2
            elif rest[j] == "block":
                out.append({"0": "conv", "1": "norm"}[rest[j + 1]]); j += 2
            elif rest[j] == "cond_encoder":
                out.append("cond_encoder"); j += 2           # Sequential(Mish, Linear, Rearrange): the Linear is index 1
            elif rest[j] == "residual_conv":
                out.append("residual_conv"); j += 1
            elif rest[j] == "conv":                          # Downsample1d / Upsample1d wrap a single conv
                j += 1
            else:
                raise KeyError(key)
        node = params
        for p in out:
            node = node.setdefault(p, {})
        node[leaf] = jnp.asarray(value.detach().cpu().numpy())
    return {"params": params}


def same_tree(a, b):
    paths = lambda t: {jax.tree_util.keystr(p): x.shape for p, x in jax.tree_util.tree_flatten_with_path(t)[0]}
    assert paths(a) == paths(b)


def t2j(x):                         # (B, C, T) torch -> (B, T, C) numpy
    return np.transpose(x.detach().numpy(), (0, 2, 1))


# ---------------------------------------------------------------- network #
def test_sinusoidal_pos_emb():
    t = torch.arange(100)
    record("SinusoidalPosEmb (dim 256, t = 0..99)", unet.SinusoidalPosEmb(256)(jnp.arange(100)), RefPosEmb(256)(t).numpy(), 1e-5,
           "positional_embedding.py L5-17")


@pytest.mark.parametrize("name, ref_module, ours, in_ch", [
    ("Conv1dBlock (16->32, k5, 8 groups)", lambda: ref_conv.Conv1dBlock(16, 32, 5, n_groups=8), lambda: unet.Conv1dBlock(32, 5, 8), 16),
    ("Downsample1d (32)", lambda: ref_conv.Downsample1d(32), lambda: unet.Conv1d(32, 3, stride=2, padding=1), 32),
    ("Upsample1d (32), ConvTranspose1d k4 s2 p1", lambda: ref_conv.Upsample1d(32), lambda: unet.ConvTranspose1d(32, 4, stride=2, padding=1), 32),
])
def test_conv_components(name, ref_module, ours, in_ch):
    ref = ref_module()
    x = torch.randn(4, in_ch, 16)
    params = torch_to_flax(ref.state_dict())
    module = ours()
    same_tree(params, module.init(jax.random.PRNGKey(0), jnp.asarray(t2j(x))))
    record(name, module.apply(params, jnp.asarray(t2j(x))), t2j(ref(x)), 1e-5, "conv1d_components.py L7-40")


@pytest.mark.parametrize("cond_predict_scale", [True, False])
def test_conditional_residual_block(cond_predict_scale):
    ref = ref_unet.ConditionalResidualBlock1D(64, 128, cond_dim=40, kernel_size=5, n_groups=8, cond_predict_scale=cond_predict_scale)
    x, cond = torch.randn(4, 64, 16), torch.randn(4, 40)
    params = torch_to_flax(ref.state_dict())
    ours = unet.ConditionalResidualBlock1D(128, 5, 8, cond_predict_scale)
    same_tree(params, ours.init(jax.random.PRNGKey(0), jnp.asarray(t2j(x)), jnp.asarray(cond.numpy())))
    record(f"ConditionalResidualBlock1D (64->128, FiLM scale={cond_predict_scale})",
           ours.apply(params, jnp.asarray(t2j(x)), jnp.asarray(cond.numpy())), t2j(ref(x, cond)), 1e-5,
           "conditional_unet1d.py L14-66")


def full_unet(obs_dim=212, action_dim=2, n_obs_steps=2):
    """The lowdim default: down_dims 256/512/1024, kernel 5, 8 groups, step embedding 256, FiLM with scale."""
    ref = ref_unet.ConditionalUnet1D(input_dim=action_dim, global_cond_dim=obs_dim * n_obs_steps, diffusion_step_embed_dim=256,
                                     down_dims=[256, 512, 1024], kernel_size=5, n_groups=8, cond_predict_scale=True)
    ours = unet.ConditionalUnet1D(input_dim=action_dim, down_dims=(256, 512, 1024), kernel_size=5, n_groups=8,
                                  diffusion_step_embed_dim=256, cond_predict_scale=True)
    return ref, ours, torch_to_flax(ref.state_dict())


def test_conditional_unet1d():
    ref, ours, params = full_unet()
    x, t, cond = torch.randn(4, 16, 2), torch.tensor([0, 17, 63, 99]), torch.randn(4, 424)
    same_tree(params, ours.init(jax.random.PRNGKey(0), jnp.asarray(x.numpy()), jnp.asarray(t.numpy()), jnp.asarray(cond.numpy())))
    n_params = sum(p.numel() for p in ref.parameters())
    record(f"ConditionalUnet1D, full lowdim config ({n_params / 1e6:.2f} M parameters)",
           ours.apply(params, jnp.asarray(x.numpy()), jnp.asarray(t.numpy()), jnp.asarray(cond.numpy())),
           ref(x, t, global_cond=cond).detach().numpy(), 1e-4, "conditional_unet1d.py L69-241")


# ---------------------------------------------------------------- DDPM #
def official_ddpm():
    return DDPMScheduler(num_train_timesteps=100, beta_start=0.0001, beta_end=0.02, beta_schedule="squaredcos_cap_v2",
                         variance_type="fixed_small", clip_sample=True, prediction_type="epsilon")


def test_ddpm_schedule():
    ref, ours = official_ddpm(), scheduler.make_ddpm(100, "squaredcos_cap_v2")
    record("DDPM betas (squaredcos_cap_v2, 100 steps)", ours.betas, ref.betas.numpy(), 1e-7)
    record("DDPM alphas_cumprod", ours.alphas_cumprod, ref.alphas_cumprod.numpy(), 1e-6)
    ref.set_timesteps(100)
    record("DDPM inference timesteps", scheduler.inference_timesteps(100, 100), ref.timesteps.numpy(), 0)


def test_ddpm_add_noise():
    ref, ours = official_ddpm(), scheduler.make_ddpm(100, "squaredcos_cap_v2")
    x0, noise, t = torch.randn(8, 16, 2), torch.randn(8, 16, 2), torch.randint(0, 100, (8,))
    record("DDPM add_noise", scheduler.add_noise(ours, jnp.asarray(x0.numpy()), jnp.asarray(noise.numpy()), jnp.asarray(t.numpy())),
           ref.add_noise(x0, noise, t).numpy(), 1e-6)


def step_diffusers_0111(s, model_output, t, sample, noise):
    """DDPMScheduler.step of diffusers 0.11.1 (the version the official code pins), transcribed from
    diffusers/schedulers/scheduling_ddpm.py L273-321 for epsilon prediction, clip_sample and fixed_small variance."""
    alpha_prod_t = s.alphas_cumprod[t]
    alpha_prod_t_prev = s.alphas_cumprod[t - 1] if t > 0 else s.one
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
    pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * s.betas[t]) / beta_prod_t
    current_sample_coeff = s.alphas[t] ** (0.5) * beta_prod_t_prev / beta_prod_t
    pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample
    if t > 0:
        variance = torch.clamp((1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * s.betas[t], min=1e-20)
        pred_prev_sample = pred_prev_sample + variance ** 0.5 * noise
    return pred_prev_sample


@pytest.mark.parametrize("t", [99, 50, 1, 0])
def test_ddpm_step(t, monkeypatch):
    ref, ours = official_ddpm(), scheduler.make_ddpm(100, "squaredcos_cap_v2")
    ref.set_timesteps(100)
    eps, sample, noise = torch.randn(8, 16, 2), torch.randn(8, 16, 2) * 2, torch.randn(8, 16, 2)
    got = scheduler.step(ours, jnp.asarray(eps.numpy()), jnp.asarray(t), jnp.asarray(sample.numpy()), jnp.asarray(noise.numpy()))
    record(f"DDPM step t={t} vs diffusers 0.11.1 formula", got, step_diffusers_0111(ref, eps, t, sample, noise).numpy(), 1e-6)
    monkeypatch.setattr(diffusers_ddpm, "randn_tensor", lambda *a, **k: noise)
    expected = ref.step(eps, t, sample).prev_sample.numpy()
    # diffusers >= 0.12 computes beta_t as 1 - alpha_bar_t / alpha_bar_{t-1} instead of reading betas[t]; identical in
    # exact arithmetic, but at t = 0 (beta_0 ~ 4e-5) the float32 rounding of 1 - alphas[0] shows up at ~1e-5
    record(f"DDPM step t={t} vs installed diffusers {diffusers.__version__}", got, expected, 1e-5 if t > 0 else 3e-5,
           "differs from 0.11.1 only in float32 rounding of beta_0" if t == 0 else "")


# ---------------------------------------------------------------- normaliser, windows #
def test_normalizer():
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(500, 12)).astype(np.float32) * rng.uniform(0.1, 5, 12).astype(np.float32)
    obs[:, 3] = 0.25                                    # a constant dimension: shifted only
    action = rng.uniform(-0.8, 0.8, (500, 2)).astype(np.float32)
    ref = LinearNormalizer()
    ref.fit({"obs": obs, "action": action}, last_n_dims=1, mode="limits")
    for key, x in (("obs", obs), ("action", action)):
        p = fit_limits(x)
        record(f"normalizer scale ({key}, limits)", p["scale"], ref.params_dict[key]["scale"].detach().numpy(), 1e-6)
        record(f"normalizer offset ({key}, limits)", p["offset"], ref.params_dict[key]["offset"].detach().numpy(), 1e-6)
        record(f"normalize ({key})", normalize(p, x), ref[key].normalize(torch.from_numpy(x)).numpy(), 1e-5)
        z = rng.uniform(-1, 1, x.shape).astype(np.float32)
        record(f"unnormalize ({key})", unnormalize(p, z), ref[key].unnormalize(torch.from_numpy(z)).numpy(), 1e-5)


class _Buffer:
    """The few ReplayBuffer members SequenceSampler uses."""
    def __init__(self, data, episode_ends):
        self.data, self.episode_ends = data, np.asarray(episode_ends)

    def keys(self):
        return self.data.keys()

    def __getitem__(self, k):
        return self.data[k]


def test_sequence_windows():
    lengths = [5, 19, 30]                               # includes an episode shorter than the window
    ends = np.cumsum(lengths)
    data = {"obs": np.arange(ends[-1] * 3, dtype=np.float32).reshape(-1, 3), "action": np.arange(ends[-1] * 2, dtype=np.float32).reshape(-1, 2)}
    ref = SequenceSampler(_Buffer(data, ends), sequence_length=16, pad_before=1, pad_after=7)
    ours = create_indices(ends, 16, pad_before=1, pad_after=7)
    record("window indices (create_indices, horizon 16, pad 1/7)", ours, ref.indices, 0)
    rows = gather_index(ours, 16)
    for key in ("obs", "action"):
        record(f"window contents with edge padding ({key})", data[key][rows],
               np.stack([ref.sample_sequence(i)[key] for i in range(len(ref.indices))]), 0)


# ---------------------------------------------------------------- policy: loss and sampling #
def official_policy(ref_net, obs, action):
    pol = DiffusionUnetLowdimPolicy(model=ref_net, noise_scheduler=official_ddpm(), horizon=16, obs_dim=obs.shape[-1],
                                    action_dim=action.shape[-1], n_action_steps=8, n_obs_steps=2, num_inference_steps=100,
                                    obs_as_global_cond=True, pred_action_steps_only=False, oa_step_convention=True)
    norm = LinearNormalizer()
    norm.fit({"obs": obs, "action": action}, last_n_dims=1, mode="limits")
    pol.set_normalizer(norm)
    return pol


def small_unet(obs_dim, action_dim):
    ref = ref_unet.ConditionalUnet1D(input_dim=action_dim, global_cond_dim=obs_dim * 2, diffusion_step_embed_dim=32,
                                     down_dims=[32, 64, 128], kernel_size=5, n_groups=8, cond_predict_scale=True)
    ours = unet.ConditionalUnet1D(input_dim=action_dim, down_dims=(32, 64, 128), kernel_size=5, n_groups=8,
                                  diffusion_step_embed_dim=32, cond_predict_scale=True)
    return ref, ours, torch_to_flax(ref.state_dict())


def toy_data(n=400, obs_dim=10, action_dim=2):
    rng = np.random.default_rng(1)
    return rng.normal(size=(n, obs_dim)).astype(np.float32), rng.uniform(-0.8, 0.8, (n, action_dim)).astype(np.float32)


def test_compute_loss(monkeypatch):
    obs, action = toy_data()
    ref_net, ours, params = small_unet(10, 2)
    pol = official_policy(ref_net, obs, action)
    batch = {"obs": torch.from_numpy(obs[:64].reshape(4, 16, 10)), "action": torch.from_numpy(action[:64].reshape(4, 16, 2))}
    noise, timesteps = torch.randn(4, 16, 2), torch.tensor([3, 40, 77, 99])
    monkeypatch.setattr(torch, "randn", lambda *a, **k: noise)
    monkeypatch.setattr(torch, "randint", lambda *a, **k: timesteps)
    expected = pol.compute_loss(batch).item()
    monkeypatch.undo()
    p_obs, p_act = fit_limits(obs), fit_limits(action)
    got = dp.compute_loss(ours, params, scheduler.make_ddpm(100), normalize(p_obs, jnp.asarray(batch["obs"].numpy())),
                          normalize(p_act, jnp.asarray(batch["action"].numpy())), jnp.asarray(noise.numpy()), jnp.asarray(timesteps.numpy()), 2)
    record("training loss (compute_loss)", got, expected, 1e-6, "diffusion_unet_lowdim_policy.py L183-252")


def test_predict_action(monkeypatch):
    """Whole inference path: normalise the observation history, 100 reverse DDPM steps, unnormalise, take the
    8 actions from index n_obs_steps - 1. Same initial noise and per-step noise on both sides."""
    obs, action = toy_data()
    ref_net, ours, params = small_unet(10, 2)
    pol = official_policy(ref_net, obs, action)
    history = torch.from_numpy(obs[:6].reshape(3, 2, 10))
    init_noise = torch.randn(3, 16, 2)
    step_noises = torch.randn(100, 3, 16, 2)
    it = iter(step_noises)
    monkeypatch.setattr(torch, "randn", lambda *a, **k: init_noise)
    monkeypatch.setattr(diffusers_ddpm, "randn_tensor", lambda *a, **k: next(it))
    with torch.no_grad():
        expected = pol.predict_action({"obs": history})["action"].numpy()
    monkeypatch.undo()

    ddpm, ts = scheduler.make_ddpm(100), jnp.asarray(scheduler.inference_timesteps(100, 100))
    p_obs, p_act = fit_limits(obs), fit_limits(action)
    cond = dp.global_cond(normalize(p_obs, jnp.asarray(history.numpy())), 2)
    naction = dp.conditional_sample(ours, params, ddpm, ts, cond, jnp.asarray(init_noise.numpy()), jnp.asarray(step_noises.numpy()))
    got = unnormalize(p_act, naction)[:, 1:9]
    record("predict_action (100 DDPM steps, 8 actions from index 1)", got, expected, 1e-4,
           "diffusion_unet_lowdim_policy.py L59-177")


# ---------------------------------------------------------------- EMA, optimiser, learning rate #
def test_ema():
    net = torch.nn.Linear(3, 2)
    ema = EMAModel(torch.nn.Linear(3, 2), update_after_step=0, inv_gamma=1.0, power=0.75, min_value=0.0, max_value=0.9999)
    steps = np.arange(0, 300000, 997)
    record("EMA decay schedule (power 0.75)", [float(dp.ema_decay(s)) for s in steps], [ema.get_decay(int(s)) for s in steps], 1e-6)
    ema.averaged_model.load_state_dict(net.state_dict())
    params = {k: jnp.asarray(v.detach().numpy()) for k, v in net.state_dict().items()}
    ema_params, n = dict(params), 0
    for _ in range(5):                                   # change the weights, then update the EMA, five times
        with torch.no_grad():
            for p in net.parameters():
                p.add_(torch.randn_like(p))
        params = {k: jnp.asarray(v.detach().numpy()) for k, v in net.state_dict().items()}
        ema_params = dp.ema_update(ema_params, params, dp.ema_decay(n))
        ema.step(net)
        n += 1
    for k in ("weight", "bias"):
        record(f"EMA weights after 5 updates ({k})", ema_params[k], ema.averaged_model.state_dict()[k].numpy(), 1e-6)


def test_adamw_with_cosine_warmup():
    """20 optimizer steps of AdamW (betas 0.95/0.999, wd 1e-6) with the cosine schedule (5 warm-up of 20 steps)."""
    rng = np.random.default_rng(2)
    w0 = rng.normal(size=(6, 4)).astype(np.float32)
    grads = rng.normal(size=(20, 6, 4)).astype(np.float32)
    w = torch.nn.Parameter(torch.from_numpy(w0.copy()))
    opt = torch.optim.AdamW([w], lr=1e-1, betas=(0.95, 0.999), eps=1e-8, weight_decay=1e-6)
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=5, num_training_steps=20, last_epoch=-1)
    lrs = []
    for g in grads:
        lrs.append(sched.get_last_lr()[0])
        w.grad = torch.from_numpy(g.copy())
        opt.step(); opt.zero_grad(); sched.step()
    schedule = dp.cosine_with_warmup(1e-1, 5, 20)
    tx = dp.make_optimizer(schedule, (0.95, 0.999), 1e-8, 1e-6)
    p = jnp.asarray(w0); state = tx.init(p)
    for g in grads:
        updates, state = tx.update(jnp.asarray(g), state, p)
        p = optax.apply_updates(p, updates)
    record("learning rate, cosine with 5 warm-up of 20 steps", [float(schedule(i)) for i in range(20)], lrs, 1e-7)
    # lr 0.1 to make differences visible. The formulas are identical (with JAX_ENABLE_X64 both sides agree to 1e-8);
    # in float32, optax computes the bias correction 1 - 0.999**t in float32 (0.999 is not exact in float32), a
    # relative error ~1e-5 in the first steps, while torch uses Python floats. At lr 1e-4 this is negligible.
    record("AdamW weights after 20 steps (lr 0.1)", p, w.detach().numpy(), 1e-5, "float32 bias-correction rounding, see comment")
