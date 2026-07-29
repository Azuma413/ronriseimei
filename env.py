from typing import NamedTuple
import jax
import jax.numpy as jnp

class CartPoleParams(NamedTuple):
    """物理パラメータ"""
    gravity: float = 9.8       # 重力加速度 [m/s^2]
    cart_mass: float = 1.0     # カートの質量 [kg]
    pole_mass: float = 0.1     # ポールの質量 [kg]
    pole_length: float = 0.5   # ポールの半長 [m]
    force_mag: float = 10.0    # 行動により加える力の大きさ [N]
    dt: float = 0.02           # 積分の時間刻み [s]
    x_limit: float = 2.4       # カート位置の許容範囲 [m]
    theta_limit: float = 12.0 * jnp.pi / 180.0  # ポール角度の許容範囲 [rad]

class TaskConfig(NamedTuple):
    """初期状態・報酬・終端条件"""
    initial_state: tuple = (0.0, 0.0, 0.0, 0.0)
    wrap_angle: bool = False
    terminate_on_limits: bool = True   # |x|>x_limit または |theta|>theta_limit で終端
    terminate_on_rail: bool = False    # レール端 |x|>x_limit のみで終端
    uprightness_reward: bool = False   # r = (1+cos(theta))/2  (非負)
    cosine_reward: bool = False        # r = cos(theta)  (負値を含む. 報酬設計の失敗例)

STABILIZATION_TASK = TaskConfig()
SWINGUP_TASK = TaskConfig(initial_state=(0.0, 0.0, jnp.pi, 0.0), wrap_angle=True, terminate_on_limits=False, uprightness_reward=True)
SWINGUP_NAIVE_TASK = TaskConfig(initial_state=(0.0, 0.0, jnp.pi, 0.0), wrap_angle=True, terminate_on_limits=False, terminate_on_rail=True, cosine_reward=True)

def dynamics(params: CartPoleParams, state: jnp.ndarray, force) -> jnp.ndarray:
    """運動方程式に基づき1ステップ後の状態を返す"""
    x, x_dot, theta, theta_dot = state
    total_mass = params.cart_mass + params.pole_mass
    ml = params.pole_mass * params.pole_length
    cos_t, sin_t = jnp.cos(theta), jnp.sin(theta)
    temp = (force + ml * theta_dot**2 * sin_t) / total_mass
    theta_acc = (params.gravity * sin_t - cos_t * temp) / (params.pole_length * (4.0 / 3.0 - params.pole_mass * cos_t**2 / total_mass))
    x_acc = temp - ml * theta_acc * cos_t / total_mass
    x_dot += params.dt * x_acc
    theta_dot += params.dt * theta_acc
    return jnp.array([x + params.dt * x_dot, x_dot, theta + params.dt * theta_dot, theta_dot])

def reset(task: TaskConfig, key: jax.Array) -> jnp.ndarray:
    return jnp.array(task.initial_state) + jax.random.uniform(key, (4,), minval=-0.05, maxval=0.05)

def transition(params: CartPoleParams, task: TaskConfig, state: jnp.ndarray, force):
    next_state = dynamics(params, state, force)
    if task.wrap_angle:  # 角度を[-pi, pi)へ折り返す
        next_state = next_state.at[2].set(jnp.mod(next_state[2] + jnp.pi, 2.0 * jnp.pi) - jnp.pi)
    x, theta = next_state[0], next_state[2]
    # done判定
    if task.terminate_on_limits:
        done = (jnp.abs(x) > params.x_limit) | (jnp.abs(theta) > params.theta_limit)
    elif task.terminate_on_rail:
        done = jnp.abs(x) > params.x_limit
    else:
        done = jnp.bool_(False)
    # 報酬計算
    if task.cosine_reward:
        reward = jnp.cos(theta)
    elif task.uprightness_reward:
        reward = (1.0 + jnp.cos(theta)) / 2.0
    else:
        reward = 1.0
    return next_state, reward, done

def step(params: CartPoleParams, task: TaskConfig, state: jnp.ndarray, action) -> tuple:
    """離散行動を力へ変換して課題を1ステップ進める."""
    force = jnp.where(action == 1, params.force_mag, -params.force_mag)
    return transition(params, task, state, force)