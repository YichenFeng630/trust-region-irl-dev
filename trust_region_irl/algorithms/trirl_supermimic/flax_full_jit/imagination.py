"""Policy and critic update of SuperMimic (Liao et al., 2026, Sec. 3.3, Eq. 2-3) with TRIRL's trust region.

From real start states the current Gaussian policy (reparameterised, a = mu + sigma * eps) acts for H steps in the frozen
world model; the policy loss is the negative imagined return, differentiated through the model (BPTT), plus TRIRL's
KL(pi_theta || pi_k) and entropy terms. Per imagined trajectory:

    G = sum_t gamma^t c_t (r_t + gamma d_{t+1} T(s_{t+1})) + gamma^H c_H V(s_H),    W = sum_t gamma^t c_t
    loss = -G / (W sigma_A) + (1/W) sum_t gamma^t c_t [eta KL(pi_theta || pi_k)(s_t) - entropy_coef^2 H(pi_theta)]

c_t is 1 until the episode ends (d = success of the predicted state), T is the value of the absorbing state after the
end (trirl_ppo_spr's terminal tail divided by gamma), sigma_A the standard deviation of the lambda-return advantages of
the batch (stop-gradient), the counterpart of PPO's per-minibatch advantage normalisation. KL and entropy are evaluated
at the stop-gradient imagined states, as PPO evaluates them at fixed sampled states. The termination is hard and has no
gradient (as SHAC, NVlabs/DiffRL algorithms/shac.py); straight_through=True passes the gradient of a sigmoid
surrogate instead. The critic regresses the lambda-returns of Eq. 3 on the imagined states (no target network).
"""
import jax
import jax.numpy as jnp


def gaussian_kl(mean, logstd, old_mean, old_logstd):
    """KL(N(mean, std) || N(old_mean, old_std)), summed over the action dimensions: trirl_ppo_spr's trust-region loss."""
    std, old_std = jnp.exp(logstd), jnp.exp(old_logstd)
    maha = 0.5 * jnp.sum(((old_mean - mean) / old_std) ** 2, axis=-1)
    cov = 0.5 * jnp.sum(2.0 * (old_logstd - logstd) + (std / old_std) ** 2 - 1.0, axis=-1)
    return maha + cov


def gaussian_entropy(logstd):
    return jnp.sum(logstd + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e), axis=-1)


def lambda_returns(rewards, dones, terminal_values, next_values, gamma, lam):
    """Eq. 3 with absorbing-state tails; arrays (H, ...) over t = 0..H-1: rewards r_t, dones d_{t+1}, terminal_values
    T(s_{t+1}), next_values V(s_{t+1}) (the last one is V(s_H)).
        R_t = r_t + gamma [d T + (1 - d)((1 - lam) V_{t+1} + lam R_{t+1})],   R_H = V_H"""
    def body(R_next, x):
        r, d, T, v_next = x
        R = r + gamma * (d * T + (1.0 - d) * ((1.0 - lam) * v_next + lam * R_next))
        return R, R
    _, R = jax.lax.scan(body, next_values[-1], (rewards, dones, terminal_values, next_values), reverse=True)
    return R


def _policy_dist(policy, params, obs):
    mean, logstd = policy.apply(params, obs)
    return mean, jnp.broadcast_to(jnp.reshape(logstd, (-1,))[-mean.shape[-1]:], mean.shape)


def make_actor_loss(policy, critic, model_step, obs_fn, reward_fn, terminal_value_fn, success_fn, soft_success_fn,
                    horizon, gamma, lam, entropy_coef, straight_through=False, st_temperature=0.1,
                    return_normalization="advantage_std", remat=True):
    """Returns (actor_loss, imagined_return).
    actor_loss(policy_params, old_policy_params, critic_params, wm_params, norm, reward_params, s0, noise, eta, threshold)
    -> (loss, aux) for a batch: s0 (B, d), noise (B, H, a).
    imagined_return(policy_params, wm_params, norm, reward_params, s0, noise, threshold) -> (B,): the H-step return
    without the critic's tail, to compare with the real return from the same start states.

    model_step(wm_params, norm, s, a) -> s'; reward_fn(reward_params, s, a, s') -> r; terminal_value_fn(reward_params, s')
    -> T; success_fn(s', threshold) -> {0, 1}; soft_success_fn(s', threshold, temperature) -> (0, 1); all single-sample."""

    def trajectory(policy_params, old_policy_params, wm_params, norm, reward_params, s0, noise, eta, threshold):
        def step(carry, eps):
            s, c, disc = carry
            mean, logstd = _policy_dist(policy, policy_params, obs_fn(s))
            a = mean + jnp.exp(logstd) * eps
            s_sg = jax.lax.stop_gradient(s)
            mean_r, logstd_r = _policy_dist(policy, policy_params, obs_fn(s_sg))
            old_mean, old_logstd = jax.lax.stop_gradient(_policy_dist(policy, old_policy_params, obs_fn(s_sg)))
            kl = gaussian_kl(mean_r, logstd_r, old_mean, old_logstd)
            ent = gaussian_entropy(logstd_r)

            s_next = model_step(wm_params, norm, s, a)
            r = reward_fn(reward_params, s, a, s_next)
            d_hard = jax.lax.stop_gradient(success_fn(s_next, threshold))
            if straight_through:
                d_soft = soft_success_fn(s_next, threshold, st_temperature * threshold)
                d = d_hard + d_soft - jax.lax.stop_gradient(d_soft)
            else:
                d = d_hard
            T = terminal_value_fn(reward_params, s_next)

            w = jax.lax.stop_gradient(disc * c)
            ret = disc * c * (r + gamma * d * T)
            reg = w * (eta * kl - entropy_coef ** 2 * ent)
            out = {"ret": ret, "reg": reg, "w": w, "state": s_sg, "reward": jax.lax.stop_gradient(r),
                   "done": d_hard, "terminal_value": jax.lax.stop_gradient(T), "alive": jax.lax.stop_gradient(c),
                   "kl": jax.lax.stop_gradient(kl), "clipped": jnp.mean((jnp.abs(a) > 1.0).astype(jnp.float32))}
            return (s_next, c * (1.0 - d), disc * gamma), out

        if remat:
            step = jax.checkpoint(step)
        (s_H, c_H, disc_H), out = jax.lax.scan(step, (s0, jnp.ones(()), jnp.ones(())), noise)
        return jnp.sum(out["ret"]), out, s_H, c_H, disc_H

    batched = jax.vmap(trajectory, in_axes=(None, None, None, None, None, 0, 0, None, None))

    def actor_loss(policy_params, old_policy_params, critic_params, wm_params, norm, reward_params, s0, noise, eta, threshold):
        G_H, out, s_H, c_H, disc_H = batched(policy_params, old_policy_params, wm_params, norm, reward_params,
                                             s0, noise, eta, threshold)
        G = G_H + disc_H * c_H * critic.apply(critic_params, s_H).squeeze(-1)   # Eq. 2: the critic beyond H
        s_H, c_H = jax.lax.stop_gradient(s_H), jax.lax.stop_gradient(c_H)
        # lambda-returns and advantages on the stop-gradient imagined states (time-major for the scan)
        states = jnp.swapaxes(out["state"], 0, 1)                                     # (H, B, d)
        values = critic.apply(critic_params, states).squeeze(-1)                       # V(s_0..s_{H-1})
        v_H = critic.apply(critic_params, s_H).squeeze(-1)
        next_values = jnp.concatenate([values[1:], v_H[None]], axis=0)
        tm = lambda x: jnp.swapaxes(x, 0, 1)
        returns = jax.lax.stop_gradient(lambda_returns(tm(out["reward"]), tm(out["done"]), tm(out["terminal_value"]),
                                                       next_values, gamma, lam))
        values = jax.lax.stop_gradient(values)
        alive = tm(out["alive"])
        n_alive = jnp.maximum(jnp.sum(alive), 1.0)
        adv = returns - values
        adv_mean = jnp.sum(alive * adv) / n_alive
        sigma_A = jnp.sqrt(jnp.sum(alive * (adv - adv_mean) ** 2) / n_alive) + 1e-8
        scale = sigma_A if return_normalization == "advantage_std" else 1.0

        W = jnp.sum(out["w"], axis=1)
        reg = jnp.sum(out["reg"], axis=1)
        loss = jnp.mean(-G / (W * scale)) + jnp.mean(reg / W)
        aux = {
            "critic_states": states, "critic_targets": returns, "critic_weights": alive,
            "metrics": {
                "imagination/return": jnp.mean(G),
                "imagination/return_per_step": jnp.mean(G / W),
                "imagination/reward": jnp.sum(alive * tm(out["reward"])) / n_alive,
                "imagination/kl": jnp.sum(alive * tm(out["kl"])) / n_alive,
                "imagination/ended_within_horizon": 1.0 - jnp.mean(c_H),
                "imagination/action_clipped": jnp.mean(out["clipped"]),
                "imagination/advantage_std": sigma_A,
                "imagination/value": jnp.sum(alive * values) / n_alive,
            },
        }
        return loss, aux

    def imagined_return(policy_params, wm_params, norm, reward_params, s0, noise, threshold):
        G_H, out, _, c_H, _ = batched(policy_params, policy_params, wm_params, norm, reward_params, s0, noise, 0.0, threshold)
        return G_H, 1.0 - c_H

    return actor_loss, imagined_return


def critic_loss(critic, critic_params, states, targets, weights):
    """Weighted squared error of V on the lambda-returns; states (n, d), targets (n,), weights (n,)."""
    v = critic.apply(critic_params, states).squeeze(-1)
    return jnp.sum(weights * (v - targets) ** 2) / jnp.maximum(jnp.sum(weights), 1.0)
