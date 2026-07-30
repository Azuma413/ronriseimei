"""方策勾配法エージェント (課題2.2節) と, 内部モデルを微分するBPTT型の解析的方策勾配"""
from functools import partial
from typing import NamedTuple
import jax
import jax.numpy as jnp
import env

# C = diag(1/2.4, 1/2, 180/(12 pi), 1/1.5): 状態の各次元を [-1, 1] へ正規化する
NORMALIZER = jnp.array([1.0 / 2.4, 1.0 / 2.0, 180.0 / (12.0 * jnp.pi), 1.0 / 1.5])

class AgentConfig(NamedTuple):
    gamma: float = 0.95        # 割引率
    alpha: float = 0.1         # 学習率
    angle_tol: float = 3e-3    # 方策改善を行うか判定する勾配推定値の角度基準 [rad]
    sigma_offset: float = 0.1  # sigma = 0.1 + 1/(1 + exp(eta))
    clip: float = 0.0          # 方策勾配のノルムの上限 (0なら制限しない)

class TrainingConfig(NamedTuple):
    total_steps: int              # 実環境で消費するステップ数
    max_episode_steps: int = 200
    num_envs: int = 64            # 並列に進める実環境の本数.
    # 方策改善のたびに全レーンをリセットするため, 1件の勾配推定値に入るエピソード数は
    # 実質的にこの値で決まる. すなわち num_envs がそのまま推定値のバッチサイズになる.

class BpttConfig(NamedTuple):
    alpha: float = 0.01   # 学習率
    horizon: int = 200    # 模擬エピソードの長さ
    batch: int = 64       # 1回の勾配計算に用いる模擬エピソード数
    updates: int = 2000   # 勾配上昇の回数
    clip: float = 0.0     # 勾配ノルムの上限 (0なら制限しない)

class PolicySpec(NamedTuple):
    """方策の状態特徴. 既定値は課題2.2節の設定 (phi = C s) そのものである.

    方策の関数形はいずれの場合も課題指定どおりの線形ガウス方策 mu = theta' phi(s) であり,
    変わるのは特徴 phi の作り方だけである. swing-upでは角度平面 (y, y_dot) をRBFで覆う
    (表形式Q学習における状態離散化の連続版に当たる).
    """
    rbf_angle: int = 0   # 角度方向のRBF中心の数 (0なら課題指定の phi = C s)
    rbf_rate: int = 0    # 角速度方向のRBF中心の数
    rate_limit: float = 8.0  # 角速度方向にRBFを張る範囲 [rad/s]
    init_scale: float = 5.0  # theta ~ U[-init_scale, init_scale] (課題指定は5)

LINEAR = PolicySpec()

def feature_vector(spec: PolicySpec, state: jnp.ndarray) -> jnp.ndarray:
    """方策への入力 phi(s). 課題指定の設定では C s そのものである."""
    if spec.rbf_angle == 0:
        return NORMALIZER * state
    y, y_dot = state[2], state[3]
    # 角度は周期的なので cos(y - c) に基づくRBFを用いる. 幅は中心の間隔に合わせる.
    centers = jnp.linspace(-jnp.pi, jnp.pi, spec.rbf_angle, endpoint=False)
    sharpness = 2.0 / (1.0 - jnp.cos(2.0 * jnp.pi / spec.rbf_angle))
    angle = jnp.exp(sharpness * (jnp.cos(y - centers) - 1.0))
    rates = jnp.linspace(-spec.rate_limit, spec.rate_limit, spec.rbf_rate)
    width = 2.0 * spec.rate_limit / (spec.rbf_rate - 1)
    rate = jnp.exp(-((y_dot - rates) / width) ** 2)
    grid = (angle[:, None] * rate[None, :]).ravel()
    grid = grid / jnp.sum(grid)  # 正規化RBF: 特徴の大きさを状態によらず一定に保つ
    # 有界な大域特徴とRBFを併せる. 前者は倒立近傍での比例・微分フィードバックを,
    # 後者は角度平面上の大域的な振り上げの構造を表す.
    smooth = jnp.array([jnp.tanh(state[0] / 2.4), state[1] / 2.0, jnp.sin(y), jnp.cos(y), y_dot / spec.rate_limit])
    return jnp.concatenate([smooth, grid])

def feature_size(spec: PolicySpec) -> int:
    return 4 if spec.rbf_angle == 0 else 5 + spec.rbf_angle * spec.rbf_rate

def param_size(spec: PolicySpec) -> int:
    """方策パラメータの個数 (最後の1つは常に eta)"""
    return feature_size(spec) + 1

def init_policy(spec: PolicySpec, key: jax.Array) -> jnp.ndarray:
    """課題指定どおり theta ~ U[-5, 5], eta ~ U[-1, 1] で初期化する"""
    key_mean, key_eta = jax.random.split(key)
    return jnp.concatenate([
        jax.random.uniform(key_mean, (feature_size(spec),), minval=-spec.init_scale, maxval=spec.init_scale),
        jax.random.uniform(key_eta, (1,), minval=-1.0, maxval=1.0)])

def mean_action(spec: PolicySpec, policy: jnp.ndarray, state: jnp.ndarray):
    """ガウス方策の平均 mu = theta' phi(s)"""
    phi = feature_vector(spec, state)
    return jnp.dot(policy[:phi.size], phi)

def feedback_gain(policy: jnp.ndarray) -> jnp.ndarray:
    """線形方策の平均 mu = theta' C s を状態フィードバック a = k' s のゲインとして表す"""
    return policy[:4] * NORMALIZER

def action_std(config: AgentConfig, policy: jnp.ndarray):
    return config.sigma_offset + jax.nn.sigmoid(-policy[-1])  # 1/(1+exp(eta))

def log_prob(config: AgentConfig, spec: PolicySpec, policy: jnp.ndarray, state: jnp.ndarray, action):
    """ガウス方策の対数尤度 ln pi(a|s; theta, eta)"""
    std = action_std(config, policy)
    return -0.5 * jnp.log(2.0 * jnp.pi * std**2) - 0.5 * ((action - mean_action(spec, policy, state)) / std) ** 2

score = jax.grad(log_prob, argnums=2)  # [grad_theta ln pi, grad_eta ln pi]

def scores(config: AgentConfig, spec: PolicySpec, policy: jnp.ndarray,
           states: jnp.ndarray, actions: jnp.ndarray):
    """全レーンのscore (解析形. jax.gradと一致することを確認している)."""
    std = action_std(config, policy)
    phi = jax.vmap(feature_vector, in_axes=(None, 0))(spec, states)
    residual = (actions - phi @ policy[:phi.shape[1]]) / std**2
    grad_theta = residual[:, None] * phi
    sigmoid = jax.nn.sigmoid(-policy[-1])
    grad_eta = -sigmoid * (1.0 - sigmoid) * (residual**2 * std**2 - 1.0) / std  # dsigma/deta * dlnpi/dsigma
    return jnp.concatenate([grad_theta, grad_eta[:, None]], axis=1)

def sample_actions(config: AgentConfig, spec: PolicySpec, policy: jnp.ndarray,
                   states: jnp.ndarray, key: jax.Array):
    """全レーンの行動を a = mu(s) + sigma * eps としてまとめてサンプルする"""
    means = jax.vmap(mean_action, in_axes=(None, None, 0))(spec, policy, states)
    return means + action_std(config, policy) * jax.random.normal(key, (states.shape[0],))

@partial(jax.jit, static_argnums=(0, 1, 2, 3))
def train_vec(params: env.CartPoleParams, config: AgentConfig, training: TrainingConfig,
              spec: PolicySpec, key: jax.Array):
    """課題2.2節の方策勾配法による学習. num_envs個の実環境を並列に進める."""
    n_lanes, n_params = training.num_envs, param_size(spec)
    key, key_policy, key_reset = jax.random.split(key, 3)
    init_carry = (
        init_policy(spec, key_policy),     # 方策パラメータ [theta, eta]
        jnp.zeros(n_params),               # 方策勾配の推定値 Delta
        1.0,                               # Deltaに取り込んだ推定値の個数 n
        env.sample_states(params, key_reset, n_lanes),
        jnp.zeros((n_lanes, n_params)),    # 適格度 z
        jnp.zeros((n_lanes, n_params)),    # エピソード内で蓄積する勾配 delta
        (jnp.arange(n_lanes) * training.max_episode_steps) // n_lanes,  # エピソードの初期位相をずらす
        jnp.zeros(n_lanes),        # エピソードの累積報酬
        jnp.zeros(n_lanes),        # エピソードの割引収益 (最大化している目的関数そのもの)
        key
    )

    def scan_step(carry, _):
        policy, delta_avg, n, states, z, delta, steps, ep_return, ep_value, key = carry
        key, key_act, key_reset = jax.random.split(key, 3)
        actions = sample_actions(config, spec, policy, states, key_act)
        next_states, rewards, dones = jax.vmap(env.step, in_axes=(None, 0, 0))(params, states, actions)

        # 方策勾配の逐次更新: z += grad ln pi, delta += z r gamma^t
        z = z + scores(config, spec, policy, states, actions)
        delta = delta + z * (rewards * config.gamma**steps)[:, None]
        finished = dones | (steps + 1 >= training.max_episode_steps)

        # 終了したエピソードのdeltaを平均して1件の推定値とし, 移動平均Deltaへ取り込む
        count = jnp.sum(finished)
        batch = jnp.sum(jnp.where(finished[:, None], delta, 0.0), axis=0) / jnp.maximum(count, 1.0)
        delta_new = jnp.where(count > 0, ((n - 1.0) * delta_avg + batch) / n, delta_avg)
        # 前回までの推定値との角度が基準より小さければ, 収束したとみなして方策を改善する
        norms = jnp.linalg.norm(delta_avg) * jnp.linalg.norm(delta_new)
        angle = jnp.arccos(jnp.clip(jnp.dot(delta_avg, delta_new) / jnp.maximum(norms, 1e-12), -1.0, 1.0))
        improve = (count > 0) & (n >= 2.0) & (angle < config.angle_tol)
        # 勾配ノルムの上限. deltaの分布は裾が重く, 稀に外れ値がDeltaを支配して通常の
        # 数百倍の更新が入り学習が発散するため, 更新の大きさを制限する.
        scale = 1.0 if config.clip <= 0.0 else jnp.minimum(
            1.0, config.clip / jnp.maximum(jnp.linalg.norm(delta_new), 1e-12))
        policy = policy + jnp.where(improve, config.alpha * scale, 0.0) * delta_new
        n = jnp.where(improve, 1.0, jnp.where(count > 0, n + 1.0, n))

        ep_return = ep_return + rewards
        ep_value = ep_value + rewards * config.gamma**steps
        fresh = env.sample_states(params, key_reset, n_lanes)
        out = (jnp.sum(finished), jnp.sum(jnp.where(finished, ep_return, 0.0)),
               jnp.sum(jnp.where(finished, steps + 1, 0)), jnp.sum(finished & dones),
               jnp.sum(jnp.where(finished, ep_value, 0.0)), improve)
        # 方策を改善したときは進行中のエピソードも破棄する. 逐次実装では方策の更新が
        # エピソードの境界で起こるため, 古い方策で貯めた z, delta を持ち越さないようにする.
        restart = finished | improve
        carry = (policy, delta_new, n,
                 jnp.where(restart[:, None], fresh, next_states),
                 jnp.where(restart[:, None], 0.0, z),
                 jnp.where(restart[:, None], 0.0, delta),
                 jnp.where(restart, 0, steps + 1),
                 jnp.where(restart, 0.0, ep_return),
                 jnp.where(restart, 0.0, ep_value), key)
        return carry, out

    carry, out = jax.lax.scan(scan_step, init_carry, None, length=training.total_steps // training.num_envs)
    return carry[0], out  # 方策と (エピソード数, 収益, 長さ, 終端回数, 割引収益, 方策改善) の和

def imagined_return(model: env.CartPoleParams, config: AgentConfig, spec: PolicySpec,
                    policy: jnp.ndarray, x0: jnp.ndarray, noise: jnp.ndarray):
    """再パラメータ化 a = mu(s) + sigma * eps した模擬エピソードの割引収益.

    行動も遷移も policy について微分可能なので, この関数の勾配が解析的な方策勾配になる.
    """
    def body(carry, eps):
        state, alive, discount = carry
        action = mean_action(spec, policy, state) + action_std(config, policy) * eps
        next_state, reward, done = env.step(model, state, action)
        plain = reward - jnp.where(done, model.fail_penalty, 0.0)  # 終端ペナルティを除いた即時報酬
        # 終端後は状態を凍結する. 発散した状態が勾配へ流れ込むのを防ぐため
        next_state = jnp.where(alive, next_state, state)
        return ((next_state, alive & ~done, discount * config.gamma),
                (jnp.where(alive, discount * reward, 0.0), alive, jnp.where(alive, plain, 0.0)))
    _, (values, alive, plain) = jax.lax.scan(body, (x0, jnp.bool_(True), 1.0), noise)
    return jnp.sum(values), (jnp.sum(alive), jnp.sum(plain))

# 1エピソードあたりの解析的方策勾配 (pathwise推定量)
analytic_gradient = jax.value_and_grad(imagined_return, argnums=3, has_aux=True)

def likelihood_ratio_gradient(params: env.CartPoleParams, config: AgentConfig, spec: PolicySpec,
                              policy: jnp.ndarray, x0: jnp.ndarray, noise: jnp.ndarray):
    """1エピソードあたりの尤度比型の方策勾配推定値 delta = sum z r gamma^t (score function推定量).

    BPTTと同じ乱数列を用いるため, 2つの推定量を同一の軌道の上で比較できる.
    """
    def body(carry, eps):
        state, z, delta, alive, discount = carry
        action = mean_action(spec, policy, state) + action_std(config, policy) * eps
        next_state, reward, done = env.step(params, state, action)
        z = z + jnp.where(alive, scores(config, spec, policy, state[None], action[None])[0], 0.0)
        delta = delta + jnp.where(alive, z * reward * discount, 0.0)
        return (jnp.where(alive, next_state, state), z, delta, alive & ~done, discount * config.gamma), None
    zero = jnp.zeros(param_size(spec))
    carry, _ = jax.lax.scan(body, (x0, zero, zero, jnp.bool_(True), 1.0), noise)
    return carry[2]

@partial(jax.jit, static_argnums=(0, 1, 2, 3))
def train_bptt(model: env.CartPoleParams, config: AgentConfig, bptt: BpttConfig,
               spec: PolicySpec, key: jax.Array):
    """内部モデルを微分して得た解析的方策勾配による学習 (BPTT).

    実環境との相互作用を一切行わず, modelの上の模擬エピソードのみで方策を改善する.
    """
    key, key_policy = jax.random.split(key)

    def scan_step(carry, _):
        policy, key = carry
        key, key_state, key_noise = jax.random.split(key, 3)
        x0 = env.sample_states(model, key_state, bptt.batch)
        noise = jax.random.normal(key_noise, (bptt.batch, bptt.horizon))
        (value, (length, plain)), grad = jax.vmap(analytic_gradient, in_axes=(None, None, None, None, 0, 0))(
            model, config, spec, policy, x0, noise)
        step = jnp.mean(grad, axis=0)
        if bptt.clip > 0.0:  # 長いホライズンでの勾配の発散を抑える
            step = step * jnp.minimum(1.0, bptt.clip / jnp.maximum(jnp.linalg.norm(step), 1e-12))
        policy = policy + bptt.alpha * step
        return (policy, key), (jnp.mean(value), jnp.mean(length), jnp.mean(plain))

    (policy, _), out = jax.lax.scan(scan_step, (init_policy(spec, key_policy), key), None, length=bptt.updates)
    return policy, out  # 方策と (割引収益, エピソード長, 終端ペナルティを除いた収益) の推移

@partial(jax.jit, static_argnums=(0, 3))
def rollout_linear(params: env.CartPoleParams, gain: jnp.ndarray, x0: jnp.ndarray, steps: int = 3000):
    """状態フィードバック a = k's による軌道. 学習した方策の平均もLQRもこの形をとる."""
    def body(state, _):
        next_state, reward, done = env.step(params, state, jnp.dot(gain, state))
        return next_state, (state, reward, done)
    return jax.lax.scan(body, x0, None, length=steps)[1]

@partial(jax.jit, static_argnums=(0, 3, 4))
def evaluate(params: env.CartPoleParams, gain: jnp.ndarray, key: jax.Array,
             num_states: int = 256, steps: int = 1000, gamma: float = 1.0):
    """初期状態分布から決定論的方策を評価し, 維持ステップ数・累積報酬・割引収益を返す"""
    x0 = env.sample_states(params, key, num_states)
    _, rewards, dones = jax.vmap(lambda state: rollout_linear(params, gain, state, steps))(x0)
    alive = jnp.cumsum(dones, axis=1) - dones == 0  # 終端した時点までを評価対象とする
    return (jnp.sum(alive, axis=1), jnp.sum(jnp.where(alive, rewards, 0.0), axis=1),
            jnp.sum(jnp.where(alive, rewards * gamma ** jnp.arange(steps), 0.0), axis=1))

def search_linear_gain(params: env.CartPoleParams, config: AgentConfig, key: jax.Array,
                       iterations: int = 40, population: int = 256, elite: int = 32, steps: int = 200):
    """線形方策 a = k's の空間で割引収益を直接最大化する (交差エントロピー法).

    方策勾配法が得た解が, 学習の失敗ではなく目的関数そのものの最適解かどうかを切り分けるために用いる.
    """
    key, key_state = jax.random.split(key)
    x0 = jax.vmap(env.reset, in_axes=(None, 0))(params, jax.random.split(key_state, 256))
    discount = config.gamma ** jnp.arange(steps)

    def value(gain, state):
        _, rewards, dones = rollout_linear(params, gain, state, steps)
        alive = jnp.cumsum(dones) - dones == 0
        return jnp.sum(jnp.where(alive, rewards * discount, 0.0))

    score = jax.jit(jax.vmap(lambda gain: jnp.mean(jax.vmap(value, in_axes=(None, 0))(gain, x0))))
    mean, std = jnp.zeros(4), jnp.full(4, 20.0)
    for _ in range(iterations):
        key, key_pool = jax.random.split(key)
        pool = mean + std * jax.random.normal(key_pool, (population, 4))
        best = pool[jnp.argsort(score(pool))[-elite:]]
        mean, std = best.mean(axis=0), best.std(axis=0) + 1e-3
    return mean

@partial(jax.jit, static_argnums=(0, 1, 4))
def rollout_policy(params: env.CartPoleParams, spec: PolicySpec, policy: jnp.ndarray,
                   x0: jnp.ndarray, steps: int = 500):
    """方策の平均 mu(s) を用いた決定論的な軌道"""
    def body(state, _):
        next_state, reward, done = env.step(params, state, mean_action(spec, policy, state))
        return next_state, (state, reward, done)
    return jax.lax.scan(body, x0, None, length=steps)[1]

@partial(jax.jit, static_argnums=(0, 1, 4, 5))
def evaluate_policy(params: env.CartPoleParams, spec: PolicySpec, policy: jnp.ndarray,
                    key: jax.Array, num_states: int = 256, steps: int = 500):
    """初期状態分布から決定論的方策を評価し, 各試行の状態列と累積報酬を返す"""
    x0 = env.sample_states(params, key, num_states)
    states, rewards, dones = jax.vmap(lambda state: rollout_policy(params, spec, policy, state, steps))(x0)
    alive = jnp.cumsum(dones, axis=1) - dones == 0
    return states, jnp.sum(jnp.where(alive, rewards, 0.0), axis=1), alive

def linearize(params: env.CartPoleParams):
    """直立平衡点まわりで線形化し, 時定数tauで離散化した状態方程式を返す (摩擦は無視する)"""
    g, l = params.gravity, params.pole_length
    total_mass = params.cart_mass + params.pole_mass
    ml = params.pole_mass * l
    d = l * (4.0 / 3.0 - params.pole_mass / total_mass)
    A_c = jnp.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -ml * g / (total_mass * d), 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, g / d, 0.0]
    ])
    B_c = jnp.array([[0.0], [1.0 / total_mass + ml / (total_mass**2 * d)], [0.0], [-1.0 / (total_mass * d)]])
    return jnp.eye(4) + params.dt * A_c, params.dt * B_c

def solve_dare(A, B, Q, R, iterations: int = 1000):
    """離散時間Riccati方程式を不動点反復で解き, フィードバックゲインを返す"""
    def body(P, _):
        K = jnp.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        return Q + A.T @ P @ (A - B @ K), None
    P, _ = jax.lax.scan(body, Q, None, length=iterations)
    return jnp.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)

def lqr_gain(params: env.CartPoleParams, config: AgentConfig) -> jnp.ndarray:
    """課題の評価関数 sum gamma^t (s'Qs + aRa) を最小化する状態フィードバック a = k's"""
    A, B = linearize(params)
    scale = jnp.sqrt(config.gamma)  # 割引つき評価関数は (A, B) を sqrt(gamma) 倍した問題に帰着する
    K = solve_dare(scale * A, scale * B, jnp.diag(jnp.array(params.cost_state)),
                   jnp.array([[params.cost_action]]))
    return -K[0]
