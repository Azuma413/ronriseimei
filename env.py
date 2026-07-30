"""カートポール環境: 課題2.1節の設定 (摩擦つき, 連続行動, tau = 1/60 s)"""
from typing import NamedTuple
import jax
import jax.numpy as jnp

class CartPoleParams(NamedTuple):
    """物理パラメータと課題設定"""
    gravity: float = 9.8            # 重力加速度 [m/s^2]
    cart_mass: float = 1.0          # カートの質量 M [kg]
    pole_mass: float = 0.1          # ポールの質量 m [kg]
    pole_length: float = 0.5        # ポールの半長 l [m] (棒の長さは 2l = 1 m)
    cart_friction: float = 0.0005   # カートの摩擦係数 mu_c
    pole_friction: float = 0.000002 # ポールの摩擦係数 mu_p
    force_limit: float = 20.0       # 行動の大きさの上限 [N] (超過分は無視される)
    dt: float = 1.0 / 60.0          # 時定数 tau [s]
    init_std: float = 0.1           # 初期状態の標準偏差 (Sigma = diag(0.01, ...))
    # 終端条件 |x| > 2.4, |x_dot| > 2, |theta| > 12 deg, |theta_dot| > 1.5
    state_limit: tuple = (2.4, 2.0, 12.0 * jnp.pi / 180.0, 1.5)
    cost_state: tuple = (1.25, 1.0, 12.0, 0.25)  # 即時報酬の重み Q
    cost_action: float = 0.01                    # 即時報酬の重み R
    fail_penalty: float = 0.0                    # 終端時に加える報酬 (課題の設定では0)
    # 以下はswing-up課題 (5章) のための設定. 既定値は課題2の安定化課題に一致する.
    initial_mean: tuple = (0.0, 0.0, 0.0, 0.0)   # 初期状態分布の平均
    terminate_on: tuple = (True, True, True, True)  # どの状態次元の逸脱で終端するか
    upright_reward: bool = False                 # Trueなら r = (1 + cos y)/2 - c (x/x_limit)^2
    position_cost: float = 0.0                   # swing-upでカート位置に課す滑らかな費用 c
    wrap_angle: bool = False                     # Trueなら角度を [-pi, pi) へ折り返す

def dynamics(params: CartPoleParams, state: jnp.ndarray, force) -> jnp.ndarray:
    """運動方程式とEuler法による離散化 (式(4), 式(5)) に基づき1ステップ後の状態を返す"""
    x, x_dot, theta, theta_dot = state
    total_mass = params.cart_mass + params.pole_mass
    ml = params.pole_mass * params.pole_length
    sin_t, cos_t = jnp.sin(theta), jnp.cos(theta)
    friction = params.cart_friction * jnp.sign(x_dot)
    theta_acc = (
        params.gravity * sin_t
        + cos_t * (friction - force - ml * theta_dot**2 * sin_t) / total_mass
        - params.pole_friction * theta_dot / ml
    ) / (params.pole_length * (4.0 / 3.0 - params.pole_mass * cos_t**2 / total_mass))
    x_acc = (force + ml * (theta_dot**2 * sin_t - theta_acc * cos_t) - friction) / total_mass
    return jnp.array([
        x + params.dt * x_dot, x_dot + params.dt * x_acc,
        theta + params.dt * theta_dot, theta_dot + params.dt * theta_acc
    ])

def sample_states(params: CartPoleParams, key: jax.Array, count: int) -> jnp.ndarray:
    """初期状態 s ~ N(initial_mean, Sigma) をまとめてサンプルする"""
    return jnp.array(params.initial_mean) + params.init_std * jax.random.normal(key, (count, 4))

def reset(params: CartPoleParams, key: jax.Array) -> jnp.ndarray:
    return sample_states(params, key, 1)[0]

def step(params: CartPoleParams, state: jnp.ndarray, action) -> tuple:
    """連続行動を受けて1ステップ進める. 報酬は r = -s'Qs - aRa (swing-upでは (1+cos y)/2)."""
    force = jnp.clip(action, -params.force_limit, params.force_limit)
    if params.upright_reward:
        reward = (1.0 + jnp.cos(state[2])) / 2.0 - params.position_cost * (state[0] / params.state_limit[0]) ** 2
    else:
        reward = -(jnp.dot(jnp.array(params.cost_state), state**2) + params.cost_action * force**2)
    next_state = dynamics(params, state, force)
    if params.wrap_angle:
        next_state = next_state.at[2].set(jnp.mod(next_state[2] + jnp.pi, 2.0 * jnp.pi) - jnp.pi)
    outside = jnp.abs(next_state) > jnp.array(params.state_limit)
    done = jnp.any(outside & jnp.array(params.terminate_on))
    return next_state, reward + jnp.where(done, params.fail_penalty, 0.0), done
