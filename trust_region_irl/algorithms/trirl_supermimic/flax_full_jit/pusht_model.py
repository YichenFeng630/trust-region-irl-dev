"""PushT (pusht_mjx) as seen by the world model: state, observation head, success (termination) and the true reward.

State s (30) = observation (24: T position (3) and quaternion (4) relative to the goal, tool position relative to the
goal (3), arm joint positions (7) and velocities (7)) + the T's free-joint velocity (6, qvel of joint "T"). The
observation has no T velocity, so without it the next state would not follow from (s, a). The analytic heads below are
the environment's own formulas (environment.py: _goal_errors, get_reward) evaluated on the predicted state.
"""
import jax
import jax.numpy as jnp

from trust_region_irl.environments.pusht_mjx.environment import _quat_to_rotvec

OBS_DIM = 24
VEL_DIM = 6
STATE_DIM = OBS_DIM + VEL_DIM
QUAT = (3, 7)
VELOCITY_DIMS = tuple(range(OBS_DIM, STATE_DIM))
ERR_CLIP = 5.0   # PushT.ERR_CLIP


def t_velocity_slice(env):
    """qvel entries of the T's free joint (6 dofs) in the environment's model."""
    adr = int(env.mj_model.jnt_dofadr[env.mj_model.joint("T").id])
    return adr, adr + VEL_DIM


def full_state(observation, qvel, vel_slice):
    """observation (..., 24), qvel (..., nv) -> state (..., 30)."""
    a, b = vel_slice
    return jnp.concatenate([observation, qvel[..., a:b]], axis=-1)


def observation(state):
    return state[..., :OBS_DIM]


def _safe_norm(x, eps=1e-12):
    """Euclidean norm whose gradient at 0 is 0 instead of NaN (the value differs by at most sqrt(eps))."""
    return jnp.sqrt(jnp.sum(x * x) + eps)


def goal_errors(state):
    """Position and orientation error of the T (as PushT._goal_errors), single state.

    The environment computes the angle as 2 arccos(w) (via _quat_to_rotvec), whose derivative is infinite at zero
    error; 2 atan2(|v|, w) is the same angle for a unit quaternion with w >= 0 and has a finite gradient, so the
    imagined return can be differentiated near the goal. tests/test_supermimic.py compares both."""
    pos = state[0:3]
    quat = state[3:7]
    quat = jnp.where(quat[0] < 0.0, -quat, quat)
    quat = quat / jnp.maximum(jnp.linalg.norm(quat), 1e-8)
    pos_err = _safe_norm(pos)
    orn_err = 2.0 * jnp.arctan2(_safe_norm(quat[1:4]), quat[0])
    pos_err = jnp.clip(jnp.nan_to_num(pos_err, nan=ERR_CLIP, posinf=ERR_CLIP), 0.0, ERR_CLIP)
    orn_err = jnp.clip(jnp.nan_to_num(orn_err, nan=jnp.pi, posinf=jnp.pi), 0.0, jnp.pi)
    return pos_err, orn_err


def goal_errors_env(state):
    """The environment's own formula (arccos), for the tests."""
    quat = state[3:7]
    quat = jnp.where(quat[0] < 0.0, -quat, quat)
    pos_err = jnp.linalg.norm(state[0:3])
    orn_err = jnp.linalg.norm(_quat_to_rotvec(quat))
    pos_err = jnp.clip(jnp.nan_to_num(pos_err, nan=ERR_CLIP, posinf=ERR_CLIP), 0.0, ERR_CLIP)
    orn_err = jnp.clip(jnp.nan_to_num(orn_err, nan=jnp.pi, posinf=jnp.pi), 0.0, jnp.pi)
    return pos_err, orn_err


def success(state, threshold):
    """1.0 if both errors are below the threshold (the episode ends there), single state."""
    pos_err, orn_err = goal_errors(state)
    return ((pos_err < threshold) & (orn_err < threshold)).astype(jnp.float32)


def soft_success(state, threshold, temperature):
    """Smooth stand-in for success() whose gradient the straight-through termination uses."""
    pos_err, orn_err = goal_errors(state)
    return jax.nn.sigmoid((threshold - pos_err) / temperature) * jax.nn.sigmoid((threshold - orn_err) / temperature)


def env_reward(state):
    """PushT's dense reward -(pos_err + orn_err) of the state reached after the action (PushT.get_reward)."""
    pos_err, orn_err = goal_errors(state)
    return -(pos_err + orn_err)
