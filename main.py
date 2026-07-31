"""課題2 カートポールの安定化制御: 方策勾配法, BPTTによる解析的方策勾配, LQRの比較"""
from pathlib import Path
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import agent
import env

FIG_DIR = Path(__file__).resolve().parent / "fig"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = 16                     # 試行数 (それぞれ別のseed値)
EVAL_STATES, EVAL_STEPS = 256, 600   # 評価: 256個の初期状態から10秒間
PG_TRAINING = agent.TrainingConfig(total_steps=960_000_000, num_envs=4096)
SPEC_PARAMS = env.CartPoleParams()                        # 課題指定どおりの報酬
SHAPED_PARAMS = env.CartPoleParams(fail_penalty=-20.0)    # 終端ペナルティを加えた報酬
WRONG_MODEL = env.CartPoleParams(fail_penalty=-20.0, pole_mass=0.5, pole_length=0.75)  # 誤った内部モデル
CONFIG = agent.AgentConfig(alpha=0.5, clip=3.0)
BPTT = agent.BpttConfig(alpha=0.1, updates=2000, batch=64, horizon=200)

# --- 5章 swing-up課題 ---
SWINGUP_PARAMS = env.CartPoleParams(          # 真下から振り上げる. レール端でのみ終端する
    initial_mean=(0.0, 0.0, float(np.pi), 0.0), terminate_on=(True, False, False, False),
    upright_reward=True, wrap_angle=True, position_cost=0.5)
# 方策の形は課題指定どおり mu = theta' phi(s) のまま, 特徴に角度平面のRBFを加える
SWINGUP_SPEC = agent.PolicySpec(rbf_angle=8, rbf_rate=5, init_scale=1.0)
SWINGUP_CONFIG = agent.AgentConfig(gamma=0.995, alpha=0.5, clip=3.0)  # 角度基準は課題指定の 3/1000 のまま
SWINGUP_BPTT = agent.BpttConfig(alpha=0.3, updates=12000, batch=64, horizon=400, clip=1.0)
SWINGUP_TRAINING = agent.TrainingConfig(total_steps=8_000_000_000, max_episode_steps=500, num_envs=4096)
SWINGUP_STEPS = 500                           # 評価するエピソード長 (8.33秒)

def save_figure(filename):
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=150)
    plt.close()

def train_policy_gradient(params, training=PG_TRAINING, config=CONFIG, spec=agent.LINEAR):
    """全seedの方策勾配法をvmapで同時に学習する"""
    run = jax.jit(jax.vmap(lambda key: agent.train_vec(params, config, training, spec, key)))
    policies, out = run(jax.random.split(jax.random.PRNGKey(0), SEEDS))
    count, returns, lengths, failures, values, improved = (np.array(value) for value in out)
    # 報酬設計を跨いで比べられるよう, 収益からは終端ペナルティを除いておく
    return np.array(policies), count, returns - params.fail_penalty * failures, lengths, values, improved

def train_analytic(model, bptt=BPTT, config=CONFIG, spec=agent.LINEAR):
    """全seedのBPTTによる解析的方策勾配をvmapで同時に学習する"""
    run = jax.jit(jax.vmap(lambda key: agent.train_bptt(model, config, bptt, spec, key)))
    policies, out = run(jax.random.split(jax.random.PRNGKey(0), SEEDS))
    return (np.array(policies), *(np.array(value) for value in out))

def evaluate_gains(params, gains, key=jax.random.PRNGKey(7)):
    """決定論的方策 a = k's を実環境で評価し, 維持ステップ数と累積報酬を返す.

    初期状態は N(0, Sigma) から引くため一部は最初から終端条件を満たす. これらは
    どの制御則でも維持できないため, 評価から除外する.
    """
    x0 = env.sample_states(params, key, EVAL_STATES)
    valid = np.array(jnp.all(jnp.abs(x0) <= jnp.array(params.state_limit), axis=1))
    _, alive, returns, _ = zip(*[agent.evaluate(params, agent.GAIN, jnp.asarray(gain), key,
                                                EVAL_STATES, EVAL_STEPS) for gain in gains])
    return np.array(alive).sum(axis=-1)[:, valid], np.array(returns)[:, valid]

def report_evaluation(label, params, gains):
    lengths, returns = evaluate_gains(params, gains)
    hold = lengths.mean(axis=-1) * params.dt          # 1試行あたりの平均維持時間 [s]
    success = (lengths >= EVAL_STEPS).mean(axis=-1)   # 全期間維持できた初期状態の割合
    print(f"[{label}] hold {np.median(hold):.2f} s (quartile {np.percentile(hold, 25):.2f}-{np.percentile(hold, 75):.2f}), "
          f"success rate {np.median(success):.2f} [{np.percentile(success, 25):.2f}, {np.percentile(success, 75):.2f}], "
          f"return {np.median(returns.mean(axis=-1)):.2f}")
    return hold

def curve(count, values, window):
    """(エピソード数, 1エピソードあたりの平均値) の学習曲線.

    1イテレーションあたりのエピソード数が変わるため, window本の移動和の比で平均を取る.
    """
    kernel = np.ones(window)
    return (np.cumsum(count)[window - 1:],
            np.convolve(values, kernel, "valid") / np.maximum(np.convolve(count, kernel, "valid"), 1.0))

def aggregate(curves, points=400):
    """試行をまとめた曲線. 発散した試行があっても残りが描けるようnanを無視する."""
    start, stop = max(x[0] for x, _ in curves), min(x[-1] for x, _ in curves)
    grid = np.linspace(start, stop, points)
    stacked = np.stack([np.interp(grid, x, y) for x, y in curves])
    stacked = np.where(np.isfinite(stacked), stacked, np.nan)
    # 中央線は中央値とする. 発散した試行があると平均は外れ値に支配され図の外へ出てしまう
    return grid, *np.nanpercentile(stacked, [50, 25, 75], axis=0)

def band(ax, curves, color, label):
    grid, middle, low, high = aggregate(curves)
    ax.fill_between(grid, low, high, color=color, alpha=0.25, lw=0)
    ax.plot(grid, middle, color=color, lw=2, label=label)
    return middle

def plot_reward_design(spec, shaped):
    """課題指定の報酬と終端ペナルティを加えた報酬の学習曲線"""
    _, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    window = spec[1].shape[1] // 2000
    for (_, count, returns, lengths, _, _), color, name in (
            (spec, "tab:red", "as specified"),
            (shaped, "tab:blue", "terminal penalty")):
        band(axes[0], [curve(count[s], returns[s], window) for s in range(SEEDS)], color, name)
        band(axes[1], [curve(count[s], lengths[s], window) for s in range(SEEDS)], color, name)
    axes[0].set_ylabel("return")
    axes[0].set_title("(a) return")
    axes[1].axhline(200, color="gray", ls=":", lw=1.2, label="cap")
    axes[1].set_ylabel("episode length")
    axes[1].set_title("(b) episode length")
    for ax in axes:
        ax.set_xlabel("episodes")
        ax.set_xscale("log")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    save_figure("reward_design.png")

def plot_method_comparison(shaped, analytic, lqr_value):
    """方策勾配法とBPTTを, 消費したエピソード数に対して比較する"""
    _, count, _, lengths, values, _ = shaped
    _, value_curve, length_curve, _ = analytic  # 最大化している目的関数 (割引収益) で揃える
    _, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    window = count.shape[1] // 2000
    bptt_x = np.arange(1, value_curve.shape[1] + 1) * BPTT.batch
    for ax, pg_values, bptt_values, ylabel, title in (
            (axes[0], values, value_curve, "discounted return", "(a) discounted return"),
            (axes[1], lengths, length_curve, "episode length", "(b) episode length")):
        band(ax, [curve(count[s], pg_values[s], window) for s in range(SEEDS)], "tab:blue",
             "policy gradient")
        band(ax, [(bptt_x, bptt_values[s]) for s in range(SEEDS)], "tab:green",
             "BPTT")
        ax.set_xlabel("episodes")
        ax.set_xscale("log")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[0].axhline(lqr_value, color="tab:red", ls="--", lw=1.5, label="LQR")
    axes[1].axhline(200, color="gray", ls=":", lw=1.2, label="cap")
    for ax in axes:
        ax.legend(fontsize=9, loc="best")
    save_figure("method_comparison.png")

def gradient_samples(params, policy, key, episodes, spec=agent.LINEAR):
    """同じ初期状態・同じ乱数列の上で, 2種類の勾配推定量を1エピソードずつ評価する"""
    key_state, key_noise = jax.random.split(key)
    x0 = env.sample_states(params, key_state, episodes)
    noise = jax.random.normal(key_noise, (episodes, BPTT.horizon))
    # params, CONFIG, spec は閉じ込めて静的に扱う (PolicySpecはPython分岐に使うため)
    over_episodes = lambda fn: jax.jit(jax.vmap(
        lambda p, state, eps: fn(params, CONFIG, spec, p, state, eps), in_axes=(None, 0, 0)))
    (_, (length, _)), pathwise = over_episodes(agent.analytic_gradient)(policy, x0, noise)
    likelihood = over_episodes(agent.likelihood_ratio_gradient)(policy, x0, noise)
    return np.array(pathwise), np.array(likelihood), np.array(length)

def plot_gradient_quality(params, policies, key=jax.random.PRNGKey(11), episodes=32768, batches=16):
    """勾配推定量の質を, 学習初期の方策と学習後の方策の2点で比較する"""
    _, axes = plt.subplots(1, len(policies), figsize=(5.5 * len(policies), 4.3))
    sizes = np.unique(np.geomspace(1, episodes // batches, 20).astype(int))
    for ax, (title, policy) in zip(axes, policies):
        pathwise, likelihood, length = gradient_samples(params, policy, key, episodes)
        # 基準方向: 標準誤差が十分小さい解析的勾配の平均を真の勾配とみなす
        reference = pathwise.mean(axis=0) / np.linalg.norm(pathwise.mean(axis=0))
        for samples, color, name in ((pathwise, "tab:green", "BPTT"),
                                     (likelihood, "tab:blue", "policy gradient")):
            aligned = []
            for size in sizes:  # size本のエピソードから作った推定値と真の勾配方向との一致度
                pooled = samples[: (episodes // size) * size].reshape(-1, size, 5).mean(axis=1)
                aligned.append(np.mean(pooled @ reference / np.linalg.norm(pooled, axis=1)))
            ax.plot(sizes, aligned, "o-", color=color, ms=4, label=name)
            error = np.linalg.norm(samples.std(axis=0) / np.sqrt(episodes)) / np.linalg.norm(samples.mean(axis=0))
            print(f"[gradient/{title}] {name}: relative std "
                  f"{np.linalg.norm(samples.std(axis=0)) / np.linalg.norm(samples.mean(axis=0)):8.1f}, "
                  f"standard error of the mean over {episodes} episodes {error:7.3f} x |mean|, "
                  f"cosine at 1 episode {aligned[0]:6.3f}, at {sizes[-1]} episodes {aligned[-1]:6.3f}")
        print(f"[gradient/{title}] mean episode length {length.mean():.1f} steps")
        ax.axhline(0.0, color="gray", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("episodes per estimate")
        ax.set_ylabel("cosine similarity")
        ax.set_title(f"{title} ({length.mean():.0f} steps/episode)")
        ax.set_ylim(-0.3, 1.05)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9, loc="lower right")
    save_figure("gradient_quality.png")

def plot_trajectories(params, gains):
    """学習後の各手法の軌道"""
    x0 = jnp.array([0.0, 0.0, 8.0 * jnp.pi / 180.0, 0.0])
    _, axes = plt.subplots(3, 1, figsize=(7.5, 7), sharex=True)
    for (name, gain, color) in gains:
        states, _, dones = (np.array(v) for v in
                            agent.rollout(params, agent.GAIN, jnp.asarray(gain), x0, EVAL_STEPS))
        end = np.argmax(dones) + 1 if dones.any() else len(dones)
        t = np.arange(end) * params.dt
        axes[0].plot(t, np.rad2deg(states[:end, 2]), color=color, label=name)
        axes[1].plot(t, states[:end, 0], color=color)
        axes[2].plot(t, np.clip(states[:end] @ np.asarray(gain), -params.force_limit, params.force_limit), color=color)
        if end < len(dones):
            axes[0].plot(t[-1], np.rad2deg(states[end - 1, 2]), "x", color=color, ms=9, mew=2)
    axes[0].axhline(12.0, color="gray", ls="--", lw=1)
    axes[0].axhline(-12.0, color="gray", ls="--", lw=1)
    axes[0].set_ylabel("pole angle [deg]")
    axes[1].set_ylabel("cart position [m]")
    axes[2].set_ylabel("force [N]")
    axes[2].set_xlabel("time [s]")
    axes[0].legend(fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.25)
    save_figure("trajectories.png")

def best_policy(policies, hold):
    """代表として, 評価で最も長く維持できたseedの方策を返す"""
    return jnp.asarray(policies[int(np.argmax(hold))])

def median_policy(policies, score):
    """代表として, 評価値が中央値に最も近いseedの方策を返す"""
    return jnp.asarray(policies[int(np.argmin(np.abs(np.asarray(score) - np.median(score))))])

def report_swingup(label, params, policies, spec=SWINGUP_SPEC):
    """swing-up方策を評価し, 収益・倒立到達時刻・終盤の倒立滞在率を出力する"""
    returns, residency, rotations = [], [], []
    for policy in policies:
        states, alive, episode_return, _ = agent.evaluate(
            params, spec, jnp.asarray(policy), jax.random.PRNGKey(5), 64, SWINGUP_STEPS)
        states, alive = np.array(states), np.array(alive)
        upright = (np.abs(states[:, :, 2]) < params.state_limit[2]) & alive
        returns.append(float(np.mean(episode_return)))
        residency.append(float(np.mean(upright[:, -100:])))          # 終盤100ステップの倒立滞在率
        # 後半で何周したか. 静止なら0, 大振幅の往復なら1前後, 回転していれば数周以上になる
        unwrapped = np.unwrap(np.nan_to_num(states[:, :, 2], nan=0.0, posinf=0.0, neginf=0.0), axis=1)
        rotations.append(float(np.median(np.abs(unwrapped[:, -1] - unwrapped[:, 150]) / (2.0 * np.pi))))
    returns, residency, rotations = np.array(returns), np.array(residency), np.array(rotations)
    finite = np.isfinite(returns)   # 発散した試行は統計から除く
    mean_return = float(np.mean(returns[finite]))
    print(f"[{label}] return {mean_return:.1f} +/- {np.std(returns[finite]):.1f} (mean +/- std over finite seeds), "
          f"upright residency {np.mean(residency):.2f} +/- {np.std(residency):.2f}, "
          f"rotations {np.median(rotations[finite]):.2f}, "
          f"finite seeds {finite.sum()}/{len(returns)}, "
          f"success (residency > 0.9) {np.mean(residency > 0.9):.0%}")
    return residency, mean_return

def plot_swingup(pg, analytic, policies, lqr_return):
    """swing-up課題の学習曲線と, 学習後の方策の軌道"""
    _, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    count, returns = pg[1], pg[2]
    window = max(count.shape[1] // 200, 1)
    band(axes[0], [curve(count[s], returns[s], window) for s in range(SEEDS)], "tab:blue",
         "policy gradient")
    bptt_x = np.arange(1, analytic[3].shape[1] + 1) * SWINGUP_BPTT.batch
    band(axes[0], [(bptt_x, analytic[3][s]) for s in range(SEEDS)], "tab:green",
         "BPTT")
    axes[0].axhline(SWINGUP_STEPS, color="gray", ls=":", lw=1.2, label="upper bound")
    axes[0].axhline(lqr_return, color="tab:red", ls="--", lw=1.5, label="LQR")
    axes[0].set_xscale("log")
    # 方策勾配法とLQRは大きな負の値へ発散するため, 0近傍を線形とする対数軸で描く
    axes[0].set_yscale("symlog", linthresh=100)
    axes[0].set_ylim(min(1.5 * lqr_return, -1000.0), SWINGUP_STEPS * 2)
    axes[0].set_xlabel("episodes")
    axes[0].set_ylabel("return")
    axes[0].set_title("(a) learning curve")
    # 評価と同じ初期状態分布からサンプルした1点を用いる (分布の平均は角速度0の特異点のため)
    x0 = env.sample_states(SWINGUP_PARAMS, jax.random.PRNGKey(5), 1)[0]
    for name, policy, spec, color in policies:  # 角度は [-180, 180] 度へ折り返して描く
        states, _, dones = agent.rollout(SWINGUP_PARAMS, spec, jnp.asarray(policy), x0, SWINGUP_STEPS)
        states, dones = np.array(states), np.array(dones)
        end = int(np.argmax(dones)) + 1 if dones.any() else SWINGUP_STEPS   # 終端後は描かない
        raw = states[:end, 2]
        turns = np.abs(np.unwrap(raw)[-1] - np.unwrap(raw)[min(150, end - 1)]) / (2.0 * np.pi)
        print(f"[swingup/trajectory] {name}: {end} steps ({'terminated' if end < SWINGUP_STEPS else 'no termination'}), "
              f"rotations {turns:.2f}")
        angle = (np.rad2deg(raw) + 180.0) % 360.0 - 180.0
        jump = np.abs(np.diff(angle)) > 180.0          # 折り返しで縦線が入るのを防ぐ
        angle = np.where(np.append(jump, False), np.nan, angle)
        t = np.arange(end) * SWINGUP_PARAMS.dt
        axes[1].plot(t, angle, color=color, lw=1.2, label=name)
        if end < SWINGUP_STEPS:                        # 終端した位置を明示する
            axes[1].plot(t[-1], angle[-1], "x", color=color, ms=8, mew=2)
    for level, style in ((0.0, "--"), (180.0, "-"), (-180.0, "-")):
        axes[1].axhline(level, color="gray", lw=0.8, ls=style)
    axes[1].set_ylim(-190, 190)
    axes[1].set_yticks([-180, -90, 0, 90, 180])
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("pole angle [deg]")
    axes[1].set_title("(b) trajectory (0 = upright)")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    save_figure("swingup.png")

if __name__ == "__main__":
    lqr = agent.lqr_gain(SHAPED_PARAMS, CONFIG)
    print("LQR gain a = k's        :", np.array2string(np.array(lqr), precision=3))
    print("equivalent policy theta :", np.array2string(np.array(lqr / agent.NORMALIZER), precision=3))

    # --- 報酬設計: 課題指定の報酬では自滅方策が最適になる ---
    for label, params in (("as specified", SPEC_PARAMS), ("terminal penalty", SHAPED_PARAMS)):
        best = agent.search_linear_gain(params, CONFIG, jax.random.PRNGKey(0))
        print(f"[CEM/{label}] best linear gain {np.array2string(np.array(best), precision=2)}")
        report_evaluation(f"CEM/{label}", params, [best])
        report_evaluation(f"LQR/{label}", params, [lqr])

    spec = train_policy_gradient(SPEC_PARAMS)
    shaped = train_policy_gradient(SHAPED_PARAMS)
    for label, (policies, count, _, _, _, improved), params in (
            ("PG/as specified", spec, SPEC_PARAMS), ("PG/terminal penalty", shaped, SHAPED_PARAMS)):
        print(f"[{label}] {count.sum(-1).mean():.0f} episodes, {improved.sum(-1).mean():.0f} policy updates")
        # ループを抜けた時点の pg_hold は終端ペナルティありの結果であり, 以降の代表方策の選択に使う
        pg_hold = report_evaluation(label, params, [agent.feedback_gain(jnp.asarray(p)) for p in policies])
    plot_reward_design(spec, shaped)

    # --- BPTTによる解析的方策勾配 ---
    analytic = train_analytic(SHAPED_PARAMS)
    analytic_hold = report_evaluation("BPTT", SHAPED_PARAMS,
                                      [agent.feedback_gain(jnp.asarray(p)) for p in analytic[0]])
    # 学習曲線と同じ条件 (200ステップ, 終端ペナルティなし) でLQRの収益を測る
    lqr_return = float(np.mean(agent.evaluate(SHAPED_PARAMS, agent.GAIN, lqr, jax.random.PRNGKey(7),
                                              EVAL_STATES, BPTT.horizon, CONFIG.gamma)[3]))
    print(f"[LQR] discounted return over {BPTT.horizon} steps: {lqr_return:.3f}")
    plot_method_comparison(shaped, analytic, lqr_return)

    pg_policies = shaped[0]
    print("PG gain    :", np.array2string(np.array(agent.feedback_gain(best_policy(pg_policies, pg_hold))), precision=3))
    print("BPTT gain  :", np.array2string(np.array(agent.feedback_gain(best_policy(analytic[0], analytic_hold))), precision=3))
    plot_gradient_quality(SHAPED_PARAMS, [
        ("initial policy", agent.init_policy(agent.LINEAR, jax.random.PRNGKey(0))),
        ("learned policy", best_policy(analytic[0], analytic_hold))])
    plot_trajectories(SHAPED_PARAMS, [
        ("policy gradient", agent.feedback_gain(best_policy(pg_policies, pg_hold)), "tab:blue"),
        ("BPTT", agent.feedback_gain(best_policy(analytic[0], analytic_hold)), "tab:green"),
        ("LQR", lqr, "tab:red")])

    # --- モデル誤差: モデルを使う2手法だけが影響を受ける ---
    wrong_analytic = train_analytic(WRONG_MODEL)
    report_evaluation("BPTT (wrong model)", SHAPED_PARAMS,
                      [agent.feedback_gain(jnp.asarray(p)) for p in wrong_analytic[0]])
    report_evaluation("LQR (wrong model)", SHAPED_PARAMS, [agent.lqr_gain(WRONG_MODEL, CONFIG)])

    # --- 5章: swing-up課題 ---
    swing_pg = train_policy_gradient(SWINGUP_PARAMS, SWINGUP_TRAINING, SWINGUP_CONFIG, SWINGUP_SPEC)
    print(f"[swingup/PG] {swing_pg[1].sum(-1).mean():.0f} episodes, {swing_pg[5].sum(-1).mean():.0f} policy updates")
    pg_residency, _ = report_swingup("swingup/PG", SWINGUP_PARAMS, swing_pg[0])
    swing_bptt = train_analytic(SWINGUP_PARAMS, SWINGUP_BPTT, SWINGUP_CONFIG, SWINGUP_SPEC)
    bptt_residency, _ = report_swingup("swingup/BPTT", SWINGUP_PARAMS, swing_bptt[0])
    # 直立点まわりで線形化したLQRは真下からでは機能しない
    lqr_policy = jnp.concatenate([lqr / agent.NORMALIZER, jnp.zeros(1)])
    _, lqr_swing_return = report_swingup("swingup/LQR", SWINGUP_PARAMS, [lqr_policy], agent.LINEAR)
    # 課題指定の特徴 phi = C s のままでは振り上げを表現できないことの確認
    plain_bptt = train_analytic(SWINGUP_PARAMS, SWINGUP_BPTT, SWINGUP_CONFIG, agent.LINEAR)
    report_swingup("swingup/BPTT (phi = Cs as specified)", SWINGUP_PARAMS, plain_bptt[0], agent.LINEAR)
    plot_swingup(swing_pg, swing_bptt, [
        ("policy gradient", median_policy(swing_pg[0], pg_residency), SWINGUP_SPEC, "tab:blue"),
        ("BPTT", median_policy(swing_bptt[0], bptt_residency), SWINGUP_SPEC, "tab:green"),
        ("LQR", lqr_policy, agent.LINEAR, "tab:red")], lqr_swing_return)
