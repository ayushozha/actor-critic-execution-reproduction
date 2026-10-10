"""Controlled simulation using Python3.6.6, NumPy1.15.1, PyTorch0.4.1 only."""
from __future__ import print_function

import argparse
import copy
import csv
import datetime
import hashlib
import json
import math
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def identity():
    assert sys.version_info[:3] == (3, 6, 6)
    assert np.__version__ == "1.15.1"
    assert torch.__version__ == "0.4.1"
    return {"executed_utc": utc(), "python": sys.version, "python_executable": sys.executable,
            "python_sha256": digest(sys.executable), "numpy": np.__version__, "torch": torch.__version__,
            "platform": platform.platform(), "processor": platform.processor(), "cuda_used": False,
            "torch_threads": torch.get_num_threads(), "runtime_manifest_sha256": digest(ROOT / "runtime" / "runtime-manifest.json")}


class Actor(nn.Module):
    def __init__(self, width):
        super(Actor, self).__init__()
        self.layers = nn.Sequential(nn.Linear(3, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1), nn.Tanh())

    def forward(self, state):
        return self.layers(state)


class Critic(nn.Module):
    def __init__(self, width):
        super(Critic, self).__init__()
        self.layers = nn.Sequential(nn.Linear(4, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1))

    def forward(self, state, action):
        return self.layers(torch.cat([state, action], 1))


class Learner(object):
    def __init__(self, cfg, algorithm, seed):
        torch.manual_seed(seed)
        self.cfg, self.algorithm, self.updates, self.actor_updates = cfg, algorithm, 0, 0
        self.actor = Actor(cfg["hidden_width"])
        self.critic = Critic(cfg["hidden_width"])
        self.actor_target, self.critic_target = copy.deepcopy(self.actor), copy.deepcopy(self.critic)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg["actor_learning_rate"])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg["critic_learning_rate"])
        if algorithm == "td3":
            self.critic2 = Critic(cfg["hidden_width"])
            self.critic2_target = copy.deepcopy(self.critic2)
            self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=cfg["critic_learning_rate"])

    def choose(self, states):
        with torch.no_grad():
            return self.actor(torch.from_numpy(np.asarray(states, dtype=np.float32))).numpy()

    def update(self, batch):
        states, actions, rewards, next_states, dones = [torch.from_numpy(x) for x in batch]
        self.updates += 1
        with torch.no_grad():
            next_action = self.actor_target(next_states)
            target = self.critic_target(next_states, next_action)
            if self.algorithm == "td3":
                noise = torch.randn_like(next_action).mul(self.cfg["td3_target_noise_sd"]).clamp(-self.cfg["td3_target_noise_clip"], self.cfg["td3_target_noise_clip"])
                next_action = (next_action + noise).clamp(-1.0, 1.0)
                target = torch.min(self.critic_target(next_states, next_action), self.critic2_target(next_states, next_action))
            target = rewards + self.cfg["discount"] * (1.0 - dones) * target
        loss = ((self.critic(states, actions) - target) ** 2).mean()
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()
        if self.algorithm == "td3":
            loss2 = ((self.critic2(states, actions) - target) ** 2).mean()
            self.critic2_optimizer.zero_grad()
            loss2.backward()
            self.critic2_optimizer.step()
        if self.algorithm == "ddpg" or self.updates % self.cfg["td3_policy_delay"] == 0:
            actor_loss = -self.critic(states, self.actor(states)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.actor_updates += 1
            pairs = [(self.actor_target, self.actor), (self.critic_target, self.critic)]
            if self.algorithm == "td3":
                pairs.append((self.critic2_target, self.critic2))
            for target_net, source_net in pairs:
                for target_param, param in zip(target_net.parameters(), source_net.parameters()):
                    target_param.data.mul_(1 - self.cfg["target_tau"]).add_(self.cfg["target_tau"], param.data)
        return float(loss.item())

    def save(self, path):
        checkpoint = {"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                      "actor_target": self.actor_target.state_dict(), "critic_target": self.critic_target.state_dict(),
                      "actor_optimizer": self.actor_optimizer.state_dict(), "critic_optimizer": self.critic_optimizer.state_dict(),
                      "updates": self.updates, "actor_updates": self.actor_updates, "torch_rng_state": torch.get_rng_state()}
        if self.algorithm == "td3":
            checkpoint.update(critic2=self.critic2.state_dict(), critic2_target=self.critic2_target.state_dict(), critic2_optimizer=self.critic2_optimizer.state_dict())
        torch.save(checkpoint, str(path))


def ac_fractions(cfg):
    dt = cfg["horizon"] / cfg["steps"]
    b = cfg["nominal_eta"] / dt
    risk = cfg["risk_aversion"] * cfg["sigma"] ** 2 * dt
    value = np.zeros(cfg["steps"] + 1)
    frac = np.ones(cfg["steps"] + 1)
    value[1] = b
    for n in range(2, cfg["steps"] + 1):
        c = risk + value[n - 1]
        frac[n] = c / (b + c)
        value[n] = b * c / (b + c)
    return frac, value


def fraction(action, n, cfg):
    return np.ones_like(action) if n == 1 else np.minimum(1.0, cfg["action_urgency_cap"] * (np.clip(action, -1, 1) + 1.0) / (2.0 * n))


class Environment(object):
    def __init__(self, cfg, regime, seed):
        self.cfg, self.regime = cfg, regime
        self.shock_rng = np.random.RandomState(seed + 10000)
        self.parameter_rng = np.random.RandomState(seed + 20000)

    def reset(self):
        c = self.cfg
        self.eta, self.alpha, ell = c["nominal_eta"], c["nominal_alpha"], c["nominal_liquidity"]
        if self.regime == "randomized":
            self.eta = math.exp(self.parameter_rng.uniform(*np.log(c["domain_eta_range"])))
            self.alpha = self.parameter_rng.uniform(*c["domain_alpha_range"])
            ell = self.parameter_rng.uniform(*c["domain_liquidity_range"])
        self.log_ell = math.log(ell)
        self.u = self.shock_rng.normal(0, c["liquidity_innovation_sd"] / math.sqrt(1 - c["liquidity_rho"] ** 2))
        self.x, self.n = c["initial_inventory"], c["steps"]
        self.total_impact, self.total_risk, self.maximum_trade = 0.0, 0.0, 0.0
        return self.state()

    def state(self):
        return np.array([self.n / float(self.cfg["steps"]), self.x / self.cfg["initial_inventory"], self.log_ell + self.u], dtype=np.float32)

    def step(self, action):
        dt = self.cfg["horizon"] / self.cfg["steps"]
        q = self.x * float(fraction(np.asarray(action), self.n, self.cfg))
        self.x = max(0.0, self.x - q)
        impact = self.eta * q * (q / dt) ** self.alpha / math.exp(self.log_ell + self.u)
        risk = self.cfg["risk_aversion"] * self.cfg["sigma"] ** 2 * dt * self.x ** 2
        self.total_impact += impact
        self.total_risk += risk
        self.maximum_trade = max(self.maximum_trade, q)
        self.n -= 1
        self.u = self.cfg["liquidity_rho"] * self.u + self.cfg["liquidity_innovation_sd"] * self.shock_rng.normal()
        return self.state(), -self.cfg["reward_scale"] * (impact + risk), self.n == 0


def freeze(cfg):
    torch.set_num_threads(1)
    paths = [HERE / "config.json", HERE / "PROTOCOL.md", HERE / "execution_study.py", ROOT / "runtime" / "runtime-manifest.json"]
    manifest = {"frozen_utc": utc(), "config": cfg, "runtime": identity(),
                "files": [{"path": str(path.relative_to(ROOT)), "sha256": digest(path)} for path in paths],
                "scientific_sources": ["lillicrap2016ddpg", "fujimoto2018td3", "arxiv_1403_2229", "arxiv_1412_4839"],
                "pre_run_heldout_evaluation_performed": False}
    target = HERE / "frozen-protocol.json"
    if target.exists():
        old = json.loads(target.read_text(encoding="utf-8"))
        assert old["config"] == cfg and old["files"] == manifest["files"], "Frozen protocol changed"
        return old
    write_json(target, manifest)
    return manifest


def preflight(cfg):
    torch.set_num_threads(1)
    start = utc()
    fractions, value = ac_fractions(cfg)
    dt = cfg["horizon"] / cfg["steps"]
    x, total, schedule = 1.0, 0.0, []
    for n in range(cfg["steps"], 0, -1):
        assert fractions[n] <= min(1.0, cfg["action_urgency_cap"] / n) + 1e-12
        q = x * fractions[n]
        x -= q
        schedule.append(q)
        total += cfg["nominal_eta"] * q * q / dt + cfg["risk_aversion"] * cfg["sigma"] ** 2 * dt * x * x
    assert abs(total - value[cfg["steps"]]) < 1e-12
    assert x == 0.0 and abs(sum(schedule) - 1.0) < 1e-12
    # Quadratic objective in intermediate inventories x_1,...,x_{N-1}; x_0=1,x_N=0.
    b = cfg["nominal_eta"] / dt
    risk = cfg["risk_aversion"] * cfg["sigma"] ** 2 * dt
    m = cfg["steps"] - 1
    matrix = np.diag(np.ones(m) * (2 * b + risk)) + np.diag(np.ones(m - 1) * -b, 1) + np.diag(np.ones(m - 1) * -b, -1)
    rhs = np.zeros(m)
    rhs[0] = b
    inventories = np.r_[1.0, np.linalg.solve(matrix, rhs), 0.0]
    assert np.linalg.eigvalsh(matrix).min() > 0
    assert np.max(np.abs(np.diff(inventories) + np.asarray(schedule))) < 1e-12
    twap_cost = cfg["nominal_eta"] + risk * sum((1 - float(k) / cfg["steps"]) ** 2 for k in range(1, cfg["steps"] + 1))
    assert total <= twap_cost
    for regime in cfg["regimes"]:
        env1, env2 = Environment(cfg, regime, 123), Environment(cfg, regime, 123)
        s1, s2 = env1.reset(), env2.reset()
        assert np.array_equal(s1, s2) and s1.shape == (3,)
        for n in range(cfg["steps"], 0, -1):
            s1, reward, done = env1.step(0.0)
            s2, reward2, done2 = env2.step(0.0)
            assert np.array_equal(s1, s2) and reward == reward2 and done == done2
            assert np.isfinite(reward) and reward <= 0 and env1.x >= 0
        assert env1.x == 0.0 and done
    smoke = []
    for algorithm in cfg["algorithms"]:
        learner = Learner(cfg, algorithm, 1)
        rng = np.random.RandomState(99)
        batch = [rng.normal(size=(64, 3)).astype(np.float32), rng.uniform(-1, 1, (64, 1)).astype(np.float32),
                 -rng.uniform(0, .5, (64, 1)).astype(np.float32), rng.normal(size=(64, 3)).astype(np.float32), np.zeros((64, 1), dtype=np.float32)]
        initial = learner.actor.layers[0].weight.detach().clone()
        losses = [learner.update(batch) for _ in range(4)]
        assert all(np.isfinite(losses)) and not torch.equal(initial, learner.actor.layers[0].weight)
        assert learner.actor_updates == (4 if algorithm == "ddpg" else 2)
        smoke.append({"algorithm": algorithm, "critic_losses": losses, "actor_updates": learner.actor_updates})
    report = {"started_utc": start, "finished_utc": utc(), "result": "PASS", "checks": 9,
              "ac_nominal_constant_liquidity_objective": total, "twap_nominal_constant_liquidity_objective": twap_cost,
              "ac_fractions": fractions.tolist(), "ac_trade_schedule": schedule, "smoke_updates": smoke,
              "not_research_training": "Only synthetic update and mathematical implementation checks; no tuning or held-out evaluation"}
    write_json(HERE / "preflight.json", report)
    print(json.dumps(report), flush=True)


def train(cfg, algorithm, regime, seed):
    torch.set_num_threads(1)
    frozen = freeze(cfg)
    out = HERE / "runs" / "{}_{}_seed{}".format(algorithm, regime, seed)
    out.mkdir(parents=True, exist_ok=False)
    started = utc()
    clock = time.time()
    write_json(out / "runtime-identity.json", identity())
    write_json(out / "configuration.json", {"algorithm": algorithm, "regime": regime, "seed": seed, "config": cfg,
                                             "frozen_protocol_sha256": digest(HERE / "frozen-protocol.json")})
    learner = Learner(cfg, algorithm, seed)
    env = Environment(cfg, regime, seed)
    exploration = np.random.RandomState(seed + 30000)
    replay_rng = np.random.RandomState(seed + 40000)
    capacity = cfg["replay_capacity"]
    arrays = [np.empty((capacity, 3), dtype=np.float32), np.empty((capacity, 1), dtype=np.float32),
              np.empty((capacity, 1), dtype=np.float32), np.empty((capacity, 3), dtype=np.float32), np.empty((capacity, 1), dtype=np.float32)]
    state = env.reset()
    log = (out / "training-episodes.csv").open("w", newline="")
    writer = csv.writer(log)
    writer.writerow(["episode", "transition", "eta", "alpha", "liquidity_mean", "impact_cost", "holding_risk_penalty", "objective", "maximum_trade", "last_critic_loss"])
    episode, loss = 0, None
    try:
        for step in range(1, cfg["training_transitions"] + 1):
            if step <= cfg["warmup_transitions"]:
                action = exploration.uniform(-1.0, 1.0)
            else:
                action = float(learner.choose(state[None, :])[0, 0]) + exploration.normal(0, cfg["exploration_noise_sd"])
                action = float(np.clip(action, -1, 1))
            next_state, reward, done = env.step(action)
            index = (step - 1) % capacity
            for array, value in zip(arrays, [state, [action], [reward], next_state, [float(done)]]):
                array[index] = value
            if step > cfg["warmup_transitions"]:
                indices = replay_rng.randint(0, min(step, capacity), size=cfg["batch_size"])
                loss = learner.update([array[indices] for array in arrays])
                if not np.isfinite(loss):
                    raise RuntimeError("Nonfinite critic loss at {}".format(step))
            state = next_state
            if done:
                episode += 1
                writer.writerow([episode, step, env.eta, env.alpha, math.exp(env.log_ell), env.total_impact, env.total_risk,
                                 env.total_impact + env.total_risk, env.maximum_trade, loss])
                state = env.reset()
            if step % cfg["checkpoint_intervals"] == 0:
                learner.save(out / "checkpoint_{:06d}.pt".format(step))
                log.flush()
                print("{} {} seed{} steps={} updates={} loss={:.6g} elapsed={:.1f}s".format(algorithm, regime, seed, step, learner.updates, loss, time.time() - clock), flush=True)
        log.close()
        np.savez_compressed(str(out / "replay-final.npz"), states=arrays[0], actions=arrays[1], rewards=arrays[2], next_states=arrays[3], dones=arrays[4])
        write_json(out / "rng-final.json", {"exploration": repr(exploration.get_state()), "replay_sampling": repr(replay_rng.get_state()),
                                           "liquidity": repr(env.shock_rng.get_state()), "parameters": repr(env.parameter_rng.get_state())})
        record = {"started_utc": started, "finished_utc": utc(), "elapsed_seconds": time.time() - clock, "result": "completed",
                  "algorithm": algorithm, "regime": regime, "seed": seed, "transitions": cfg["training_transitions"], "episodes": episode,
                  "critic_updates": learner.updates, "actor_updates": learner.actor_updates, "test_tuning_performed": False,
                  "code_sha256": digest(__file__), "frozen_protocol_sha256": digest(HERE / "frozen-protocol.json")}
        record["artifacts"] = [{"path": str(path.relative_to(ROOT)), "sha256": digest(path)} for path in sorted(out.iterdir()) if path.is_file()]
        write_json(out / "execution-record.json", record)
        return record
    except BaseException:
        log.close()
        write_json(out / "failure.json", {"started_utc": started, "failed_utc": utc(), "traceback": traceback.format_exc()})
        raise


def evaluation_paths(cfg):
    rng = np.random.RandomState(cfg["evaluation_seed"])
    shape = (cfg["evaluation_episodes_per_scenario"], cfg["steps"])
    u = np.empty(shape)
    initial = rng.normal(size=shape[0])
    innovations = rng.normal(size=shape)
    z = rng.normal(size=shape)
    u[:, 0] = initial * cfg["liquidity_innovation_sd"] / math.sqrt(1 - cfg["liquidity_rho"] ** 2)
    for k in range(1, cfg["steps"]):
        u[:, k] = cfg["liquidity_rho"] * u[:, k - 1] + cfg["liquidity_innovation_sd"] * innovations[:, k]
    return u, z, initial, innovations


def evaluate(cfg):
    torch.set_num_threads(1)
    freeze(cfg)
    start = utc()
    out = HERE / "evaluation"
    out.mkdir(exist_ok=False)
    u, z, initial, innovations = evaluation_paths(cfg)
    np.savez_compressed(str(out / "common-paths.npz"), log_liquidity_deviation=u, price_innovations=z, stationary_initial=initial, liquidity_innovations=innovations)
    fractions, unused = ac_fractions(cfg)
    policies = [("twap", "none", -1, None), ("ac_nominal", "none", -1, None)]
    for algorithm in cfg["algorithms"]:
        for regime in cfg["regimes"]:
            for seed in cfg["training_seeds"]:
                learner = Learner(cfg, algorithm, seed)
                checkpoint = HERE / "runs" / "{}_{}_seed{}".format(algorithm, regime, seed) / "checkpoint_{:06d}.pt".format(cfg["training_transitions"])
                learner.actor.load_state_dict(torch.load(str(checkpoint))["actor"])
                learner.actor.eval()
                policies.append((algorithm, regime, seed, learner))
    raw = (out / "raw-episodes.csv").open("w", newline="")
    writer = csv.writer(raw)
    writer.writerow(["scenario", "algorithm", "regime", "training_seed", "episode", "impact_cost", "holding_risk_penalty", "price_slippage", "implementation_shortfall", "objective", "maximum_trade", "terminal_inventory"])
    mean_rows = []
    dt = cfg["horizon"] / cfg["steps"]
    for scenario in cfg["scenarios"]:
        log_liquidity = math.log(scenario["liquidity"]) + u
        for algorithm, regime, seed, learner in policies:
            x = np.ones(u.shape[0]) * cfg["initial_inventory"]
            impact, risk, price, maximum = np.zeros(u.shape[0]), np.zeros(u.shape[0]), np.zeros(u.shape[0]), np.zeros(u.shape[0])
            schedules = np.empty_like(u)
            for k in range(cfg["steps"]):
                n = cfg["steps"] - k
                if algorithm == "twap":
                    f = np.ones_like(x) / n
                elif algorithm == "ac_nominal":
                    f = np.ones_like(x) * fractions[n]
                else:
                    states = np.column_stack([np.ones_like(x) * n / cfg["steps"], x / cfg["initial_inventory"], log_liquidity[:, k]]).astype(np.float32)
                    f = fraction(learner.choose(states)[:, 0], n, cfg)
                assert np.all(f >= 0) and np.all(f <= min(1.0, cfg["action_urgency_cap"] / n) + 1e-7)
                q = x * f
                x = np.maximum(0, x - q)
                impact += scenario["eta"] * q * (q / dt) ** scenario["alpha"] / np.exp(log_liquidity[:, k])
                risk += cfg["risk_aversion"] * cfg["sigma"] ** 2 * dt * x ** 2
                price += -cfg["sigma"] * math.sqrt(dt) * x * z[:, k]
                maximum = np.maximum(maximum, q)
                schedules[:, k] = q
            assert np.max(x) == 0.0 and np.max(np.abs(schedules.sum(axis=1) - cfg["initial_inventory"])) < 1e-6
            objective, shortfall = impact + risk, impact + price
            assert np.all(np.isfinite(objective)) and np.all(impact >= 0)
            for episode in range(len(x)):
                writer.writerow([scenario["name"], algorithm, regime, seed, episode, impact[episode], risk[episode], price[episode], shortfall[episode], objective[episode], maximum[episode], x[episode]])
            key = "{}_{}_{}".format(scenario["name"], algorithm, regime) + ("_seed{}".format(seed) if seed >= 0 else "")
            np.savez_compressed(str(out / (key + ".npz")), trades=schedules, impact=impact, risk=risk, slippage=price, objective=objective, shortfall=shortfall)
            mean_rows.append({"scenario": scenario["name"], "algorithm": algorithm, "regime": regime, "seed": seed,
                              "mean_impact": float(impact.mean()), "mean_risk": float(risk.mean()), "mean_slippage": float(price.mean()),
                              "mean_shortfall": float(shortfall.mean()), "std_shortfall": float(shortfall.std(ddof=1)),
                              "mean_objective": float(objective.mean()), "objective_standard_error_episodes": float(objective.std(ddof=1) / math.sqrt(len(x))),
                              "mean_maximum_trade": float(maximum.mean()), "terminal_inventory_max": float(x.max())})
        raw.flush()
        print("Evaluated {}: {} policies x {} common episodes".format(scenario["name"], len(policies), len(x)), flush=True)
    raw.close()
    write_json(out / "seed-means.json", mean_rows)
    record = {"started_utc": start, "finished_utc": utc(), "result": "completed", "policy_instances": len(policies), "scenarios": len(cfg["scenarios"]),
              "episodes_per_policy_scenario": len(x), "total_episode_evaluations": len(policies) * len(cfg["scenarios"]) * len(x),
              "code_sha256": digest(__file__), "frozen_protocol_sha256": digest(HERE / "frozen-protocol.json"), "test_tuning_performed": False,
              "artifacts": [{"path": str(path.relative_to(ROOT)), "sha256": digest(path)} for path in sorted(out.iterdir()) if path.is_file()]}
    write_json(out / "execution-record.json", record)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["freeze", "preflight", "train", "evaluate"])
    parser.add_argument("--algorithm", choices=["ddpg", "td3"])
    parser.add_argument("--regime", choices=["nominal", "randomized"])
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    cfg = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    if args.mode == "freeze":
        print(json.dumps(freeze(cfg), indent=2), flush=True)
    elif args.mode == "preflight":
        preflight(cfg)
    elif args.mode == "train":
        train(cfg, args.algorithm, args.regime, args.seed)
    elif args.mode == "evaluate":
        evaluate(cfg)


if __name__ == "__main__":
    main()
