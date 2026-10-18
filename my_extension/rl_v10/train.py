#!/usr/bin/env python3
"""
rl_v10 / train.py -- tabula-rasa PPO self-play in the batched simulator.

No imitation, no hand-coded behaviour: every ship move, dropoff and spawn is
sampled from the network.  The only constraints are the game's legality rules
(a ship can't move without paying the move cost, can't build where it can't
afford, can't spawn below 1000 halite).

Rewards (units of 1000 halite):
  ship   potential-shaped banked halite: r = dep + gamma*Phi' - Phi with
         Phi = cargo * (phi_far + phi_near * max(0, 1 - dist_home/phi_radius));
         over a ship's life the shaping telescopes away, so the total is exactly
         the halite it banked.  Cargo lost to a wreck or the game end costs Phi.
         Wrecks also credit destroyed enemy cargo (a*cargo) and enemy hulls.
         Building a dropoff earns its bank delta plus --lam-drop x halite
         later deposited there.  Ships alive at the end share the game outcome
         (--w-win * tanh(margin/5000)).
  spawn  per-decision return on investment: lifetime halite banked by the
         new ship - 1000.
League: each game's second seat is the current policy (mirror self-play) or a
frozen past snapshot.
"""
import argparse
import glob
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sim import BatchSim, MAX_IDS  # noqa: E402
from obs import encode, mirror_flags, uncanon_actions, N_ACT, _dist_maps  # noqa: E402
from model import Net, save, load  # noqa: E402

NEG = -1e9


def policy_step(net, obs, mask, fy, fx, greedy=False):
    """obs (n,C,H,W) np; mask (n,6,H,W) np bool. Returns per-cell actions & stats."""
    with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
        x = torch.from_numpy(obs).to(memory_format=torch.channels_last)
        lg, v, sl, sv = net(x, torch.from_numpy(fy), torch.from_numpy(fx))
    lg = lg.float()
    m = torch.from_numpy(mask)
    lg = torch.where(m, lg, torch.full_like(lg, NEG))
    lg = lg.permute(0, 2, 3, 1)                       # n,H,W,6
    if greedy:
        a = lg.argmax(-1)
    else:
        a = torch.distributions.Categorical(logits=lg).sample()
    logp = torch.log_softmax(lg, -1).gather(-1, a[..., None])[..., 0]
    sl = sl.float()
    sa = sl.argmax(-1) if greedy else torch.distributions.Categorical(logits=sl).sample()
    slogp = torch.log_softmax(sl, -1).gather(-1, sa[:, None])[:, 0]
    return a.numpy(), logp.numpy(), v.float().numpy(), sa.numpy(), slogp.numpy(), sv.float().numpy()


class League:
    def __init__(self, ckdir):
        self.ckdir = ckdir
        self.cache = {}

    def pool(self):
        return sorted(glob.glob(os.path.join(self.ckdir, 'snap_*.pt')))

    def get(self, path):
        if path not in self.cache:
            if len(self.cache) > 6:
                self.cache.pop(next(iter(self.cache)))
            net, _ = load(path)
            net = net.to(memory_format=torch.channels_last).eval()
            self.cache[path] = net
        return self.cache[path]


def phi_coef(dist, args):
    """Potential per unit cargo: grows as the ship nears one of its own structures."""
    return args.phi_far + args.phi_near * np.maximum(0.0, 1.0 - dist / args.phi_radius)


def rollout(net, league, args, rng):
    B, H, W = args.envs, 32, 32
    sim = BatchSim(B, W, H)
    seeds = [int(rng.integers(0, 2**31)) for _ in range(B)]
    init_bank = None
    if args.it <= args.curr_iters:
        frac = 1.0 - (args.it - 1) / args.curr_iters
        hi = 5000 + frac * (args.curr_max_bank - 5000)
        init_bank = np.where(rng.random(B) < 0.5, 5000, rng.uniform(5000, hi, B)).astype(np.int64)
    sim.reset(seeds, init_bank)
    mirror = mirror_flags(sim.fac, W)
    fcell = np.zeros((B, 2, 2), np.int64)                  # canonical (y,x) of each seat's yard
    for b in range(B):
        for p in (0, 1):
            fx, fy = sim.fac[b, p]
            fcell[b, p] = (fy, W - 1 - fx if mirror[b, p] else fx)
    # seat control: 'L' learner or snapshot path
    pool = league.pool()
    ctrl = []
    for b in range(B):
        if pool and rng.random() > args.p_self:
            # bias towards recent snapshots
            k = len(pool)
            w = np.arange(1, k + 1, dtype=float) ** 1.0
            opp = pool[rng.choice(k, p=w / w.sum())]
        else:
            opp = 'L'
        c = ['L', opp] if rng.random() < 0.5 else [opp, 'L']
        ctrl.append(c)
    learner = np.array([[c == 'L' for c in cc] for cc in ctrl])

    gamma, alpha = args.gamma, args.alpha
    # storage
    S_obs, S_fy, S_fx = [], [], []
    sh = {k: [] for k in ('obs', 'cell', 'act', 'logp', 'val', 'mask', 'rew', 'done', 'b', 'p', 'sid', 'turn', 'cargo_b', 'phi_b')}
    sp = {k: [] for k in ('obs', 'act', 'logp', 'val', 'b', 'p', 'sid')}
    last = np.full((B, 2, MAX_IDS), -1, np.int64)
    next_idx = []
    life_dep = np.zeros((B, MAX_IDS), np.float64)
    struct_dep = np.zeros((B, MAX_IDS), np.float64)
    n_obs = 0
    n_sh = 0
    sp_filled = 0
    stats = {'kills': 0, 'wrecks': 0, 'drops': 0, 'spawns': 0, 'self_crash': 0}

    while not sim.done.all() and sim.turn <= 400:
        turn = sim.turn
        obs, mask, spok = encode(sim.halite, sim.owner, sim.occ, sim.cargo, sim.bank, turn,
                                 sim.initial_halite, sim.fac, mirror)
        act_grid = np.zeros((B, 2, H, W), np.int64)
        spawn = np.zeros((B, 2), bool)
        live = ~sim.done
        kmap = [phi_coef(_dist_maps(sim.owner, 0), args), phi_coef(_dist_maps(sim.owner, 1), args)]
        # group seats by controller
        groups = {}
        for b in range(B):
            if not live[b]:
                continue
            for p in (0, 1):
                groups.setdefault(ctrl[b][p], []).append((b, p))
        for key, seats in groups.items():
            bs = np.array([s[0] for s in seats])
            ps = np.array([s[1] for s in seats])
            o = obs[bs, ps]
            mk = mask[bs, ps]
            fy = fcell[bs, ps, 0]
            fx = fcell[bs, ps, 1]
            model = net if key == 'L' else league.get(key)
            a, lp, v, sa, slp, sv = policy_step(model, o, mk, fy, fx)
            if key == 'L' and args.it <= args.spawn_warmup:
                sa = (rng.random(len(seats)) < args.spawn_explore).astype(np.int64)
            for i, (b, p) in enumerate(seats):
                canon_occ = mk[i, 0]
                spawn[b, p] = bool(sa[i] == 1) and bool(spok[b, p])
                act_grid[b, p] = uncanon_actions(np.where(canon_occ, a[i], 0), mirror[b, p])
                if key != 'L':
                    continue
                oi = n_obs
                n_obs += 1
                S_obs.append(o[i].astype(np.float16))
                S_fy.append(fy[i])
                S_fx.append(fx[i])
                ys, xs = np.nonzero(canon_occ)
                if len(ys):
                    # sid lookup in engine frame
                    ex = (W - 1 - xs) if mirror[b, p] else xs
                    sids = sim.sid[b, p, ys, ex]
                    k = len(ys)
                    idx = np.arange(n_sh, n_sh + k)
                    prev = last[b, p, sids]
                    for pi_, ci_ in zip(prev[prev >= 0], idx[prev >= 0]):
                        next_idx[pi_] = ci_
                    last[b, p, sids] = idx
                    next_idx.extend([-1] * k)
                    n_sh += k
                    sh['obs'].append(np.full(k, oi))
                    sh['cell'].append(ys * W + xs)
                    sh['act'].append(a[i, ys, xs])
                    sh['logp'].append(lp[i, ys, xs])
                    sh['val'].append(v[i, ys, xs])
                    sh['mask'].append(mk[i][:, ys, xs].T.copy())
                    sh['b'].append(np.full(k, b))
                    sh['sid'].append(sids)
                    sh['turn'].append(np.full(k, turn))
                    sh['cargo_b'].append(sim.cargo[b, p, ys, ex])
                    sh['phi_b'].append(sim.cargo[b, p, ys, ex] * kmap[p][b, ys, ex])
                    sh['p'].append(np.full(k, p))
                if spok[b, p] and args.it > args.spawn_warmup:
                    sp['obs'].append(oi)
                    sp['act'].append(int(sa[i]))
                    sp['logp'].append(float(slp[i]))
                    sp['val'].append(float(sv[i]))
                    sp['b'].append(b)
                    sp['p'].append(p)
                    sp['sid'].append(-1)
        ev = sim.step(act_grid, spawn)
        # bookkeeping for this turn's learner samples
        life_dep += ev['dep'] + alpha * ev['kill_cargo']
        struct_dep += ev['struct_dep']
        stats['kills'] += int(ev['kill_hulls'].sum())
        stats['wrecks'] += int(ev['dead'].sum())
        stats['drops'] += int(ev['constructed'].sum())
        stats['spawns'] += int((ev['spawned'] >= 0).sum())
        # fill spawned sid for this turn's spawn samples
        for j in range(sp_filled, len(sp['sid'])):
            b, p = sp['b'][j], sp['p'][j]
            sp['sid'][j] = int(ev['spawned'][b, p]) if sp['act'][j] == 1 and ev['spawned'][b, p] >= 0 else -2
        sp_filled = len(sp['sid'])
        cargo_after = np.full((B, MAX_IDS), -1, np.int64)
        bb, pp, yy, xx = np.nonzero(sim.occ)
        cargo_after[bb, sim.sid[bb, pp, yy, xx]] = sim.cargo[bb, pp, yy, xx]
        kmap2 = [phi_coef(_dist_maps(sim.owner, 0), args), phi_coef(_dist_maps(sim.owner, 1), args)]
        k_after = np.zeros((B, MAX_IDS))
        for p_ in (0, 1):
            sel = pp == p_
            k_after[bb[sel], sim.sid[bb[sel], p_, yy[sel], xx[sel]]] = kmap2[p_][bb[sel], yy[sel], xx[sel]]
        if len(sh['act']) and sh['turn'][-1][0] == turn:
            # samples of this turn are the trailing arrays with turn == turn
            j = len(sh['turn']) - 1
            while j >= 0 and sh['turn'][j][0] == turn:
                j -= 1
            hull = 1000.0 * min(1.0, (400 - turn) / 200.0)
            for q in range(j + 1, len(sh['turn'])):
                b = sh['b'][q]
                s = sh['sid'][q]
                cb = sh['phi_b'][q].astype(np.float64)
                bi = b[0]
                dep = ev['dep'][bi, s].astype(np.float64)
                dead = ev['dead'][bi, s]
                cons = ev['constructed'][bi, s]
                ca = cargo_after[bi, s]
                over = sim.done[bi]
                r = np.zeros(len(s))
                done = np.zeros(len(s), bool)
                # construct
                r = np.where(cons, ev['construct_delta'][bi, s] - cb, r)
                # wreck
                kv = alpha * ev['kill_cargo'][bi, s] + hull * ev['kill_hulls'][bi, s]
                r = np.where(dead & ~cons, dep + kv - cb, r)
                alive = (ca >= 0) & ~dead & ~cons
                if over:
                    r = np.where(alive, dep - cb, r)
                    done[:] = True
                else:
                    r = np.where(alive, dep + gamma * ca * k_after[bi, s] - cb, r)
                    done = dead | cons
                sh['rew'].append(r / 1000.0)
                sh['done'].append(done)
        if sim.done.all():
            break

    # terminal: outcome bonus + dropoff payoff + spawn returns
    final = sim.bank.astype(np.float64)
    cat = lambda k, dt: np.concatenate(sh[k]) if sh[k] else np.zeros(0, dt)  # noqa: E731
    rew = cat('rew', np.float64)
    done = cat('done', bool)
    b_arr = cat('b', np.int64)
    p_arr = cat('p', np.int64)
    sid_arr = cat('sid', np.int64)
    act_arr = cat('act', np.int64)
    nxt = np.array(next_idx, np.int64)
    margin = final[b_arr, p_arr] - final[b_arr, 1 - p_arr]
    end_turn = cat('turn', np.int64)
    # identify samples on the final turn of their env (ships alive at the end)
    env_last_turn = np.zeros(B, np.int64)
    for b in range(B):
        sel = b_arr == b
        env_last_turn[b] = end_turn[sel].max() if sel.any() else 0
    at_end = end_turn == env_last_turn[b_arr]
    rew = rew + np.where(at_end, args.w_win * np.tanh(margin / 5000.0), 0.0)
    cons_mask = act_arr == 5
    rew = rew + np.where(cons_mask & done, args.lam_drop * struct_dep[b_arr, np.maximum(sid_arr, 0)] / 1000.0, 0.0)

    sp_b = np.array(sp['b'], np.int64)
    sp_p = np.array(sp['p'], np.int64)
    sp_sid = np.array(sp['sid'], np.int64)
    sp_r = np.where(sp_sid >= 0, (life_dep[sp_b, np.maximum(sp_sid, 0)] - 1000.0) / 1000.0, 0.0)

    learner_scores = []
    for b in range(B):
        for p in (0, 1):
            if learner[b, p]:
                learner_scores.append((final[b, p], final[b, 1 - p], ctrl[b][1 - p]))
    stats['deposited'] = float(np.mean([sim.deposited[b, p] for b in range(B) for p in (0, 1) if learner[b, p]]))
    stats['ships_per_seat'] = float(stats['spawns'] / (2 * B))
    data = {
        'obs': np.stack(S_obs), 'fy': np.array(S_fy), 'fx': np.array(S_fx),
        'sh_obs': cat('obs', np.int64), 'sh_cell': cat('cell', np.int64),
        'sh_act': act_arr, 'sh_logp': cat('logp', np.float32),
        'sh_val': cat('val', np.float32),
        'sh_mask': np.concatenate(sh['mask']) if sh['mask'] else np.zeros((0, 6), bool),
        'sh_rew': rew, 'sh_done': done, 'sh_next': nxt,
        'sp_obs': np.array(sp['obs'], np.int64), 'sp_act': np.array(sp['act'], np.int64),
        'sp_logp': np.array(sp['logp']), 'sp_val': np.array(sp['val']), 'sp_rew': sp_r,
    }
    return data, learner_scores, stats


def gae(data, gamma, lam):
    r, v, d, nx = data['sh_rew'], data['sh_val'], data['sh_done'], data['sh_next']
    n = len(r)
    adv = np.zeros(n)
    nv = np.where(nx >= 0, v[np.maximum(nx, 0)], 0.0)
    nv = np.where(d, 0.0, nv)
    delta = r + gamma * nv - v
    # samples are stored in turn order; next always has a larger index
    for i in range(n - 1, -1, -1):
        j = nx[i]
        adv[i] = delta[i] + (gamma * lam * adv[j] if (j >= 0 and not d[i]) else 0.0)
    return adv, adv + v


def ppo_update(net, opt, data, args):
    adv, ret = gae(data, args.gamma, args.lam)
    n_obs = len(data['obs'])
    sh_obs = data['sh_obs']
    order = np.argsort(sh_obs, kind='stable')
    starts = np.searchsorted(sh_obs[order], np.arange(n_obs))
    ends = np.searchsorted(sh_obs[order], np.arange(n_obs), side='right')
    sp_of_obs = {o: i for i, o in enumerate(data['sp_obs'])}
    adv_n = (adv - adv.mean()) / (adv.std() + 1e-8) if len(adv) else adv
    sp_adv = data['sp_rew'] - data['sp_val']
    sp_adv_n = (sp_adv - sp_adv.mean()) / (sp_adv.std() + 1e-8) if len(sp_adv) > 1 else sp_adv
    W = 32
    logs = []
    for ep in range(args.epochs):
        perm = np.random.permutation(n_obs)
        for k in range(0, n_obs, args.mb):
            ob = perm[k:k + args.mb]
            si = np.concatenate([order[starts[o]:ends[o]] for o in ob]) if len(ob) else np.zeros(0, int)
            local = {o: i for i, o in enumerate(ob)}
            x = torch.from_numpy(data['obs'][ob].astype(np.float32)).to(memory_format=torch.channels_last)
            fy = torch.from_numpy(data['fy'][ob])
            fx = torch.from_numpy(data['fx'][ob])
            with torch.autocast('cpu', dtype=torch.bfloat16):
                lg, v, sl, sv = net(x, fy, fx)
            lg, v, sl, sv = lg.float(), v.float(), sl.float(), sv.float()
            loss = torch.zeros(())
            info = {}
            if len(si):
                li = torch.from_numpy(np.array([local[o] for o in sh_obs[si]]))
                cell = torch.from_numpy(data['sh_cell'][si])
                cy, cx = cell // W, cell % W
                l = lg[li, :, cy, cx]
                m = torch.from_numpy(data['sh_mask'][si])
                l = torch.where(m, l, torch.full_like(l, NEG))
                lp_all = torch.log_softmax(l, -1)
                a = torch.from_numpy(data['sh_act'][si])
                lp = lp_all.gather(1, a[:, None])[:, 0]
                old = torch.from_numpy(data['sh_logp'][si]).float()
                ratio = torch.exp(lp - old)
                A = torch.from_numpy(adv_n[si]).float()
                pg = -torch.min(ratio * A, ratio.clamp(1 - args.clip, 1 + args.clip) * A).mean()
                vpred = v[li, cy, cx]
                R = torch.from_numpy(ret[si]).float()
                vl = F.smooth_l1_loss(vpred, R)
                p_all = lp_all.exp()
                ent = -(p_all * torch.where(m, lp_all, torch.zeros_like(lp_all))).sum(1).mean()
                loss = loss + pg + args.vf * vl - args.ent * ent
                with torch.no_grad():
                    kl = (old - lp).mean().item()
                info.update(pg=pg.item(), vl=vl.item(), ent=ent.item(), kl=kl)
            spi = [sp_of_obs[o] for o in ob if o in sp_of_obs]
            if spi:
                spi = np.array(spi)
                li = torch.from_numpy(np.array([local[o] for o in data['sp_obs'][spi]]))
                slp_all = torch.log_softmax(sl[li], -1)
                a = torch.from_numpy(data['sp_act'][spi])
                lp = slp_all.gather(1, a[:, None])[:, 0]
                old = torch.from_numpy(data['sp_logp'][spi]).float()
                ratio = torch.exp(lp - old)
                A = torch.from_numpy(sp_adv_n[spi]).float()
                spg = -torch.min(ratio * A, ratio.clamp(1 - args.clip, 1 + args.clip) * A).mean()
                svl = F.smooth_l1_loss(sv[li], torch.from_numpy(data['sp_rew'][spi]).float())
                sent = -(slp_all.exp() * slp_all).sum(1).mean()
                loss = loss + spg + args.vf * svl - args.ent * sent
                info.update(spg=spg.item(), svl=svl.item(), sent=sent.item())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.max_grad)
            opt.step()
            logs.append(info)
        if logs and np.mean([d.get('kl', 0) for d in logs[-max(1, n_obs // args.mb):]]) > args.target_kl:
            break
    out = {}
    for k in logs[0] if logs else []:
        out[k] = float(np.mean([d[k] for d in logs if k in d]))
    out['adv_std'] = float(adv.std()) if len(adv) else 0.0
    out['ret_mean'] = float(ret.mean()) if len(ret) else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iters', type=int, default=100000)
    ap.add_argument('--envs', type=int, default=16)
    ap.add_argument('--ch', type=int, default=48)
    ap.add_argument('--blocks', type=int, default=6)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--mb', type=int, default=256)
    ap.add_argument('--gamma', type=float, default=0.995)
    ap.add_argument('--lam', type=float, default=0.95)
    ap.add_argument('--clip', type=float, default=0.2)
    ap.add_argument('--ent', type=float, default=0.01)
    ap.add_argument('--vf', type=float, default=0.5)
    ap.add_argument('--alpha', type=float, default=0.5, help='value of destroyed enemy cargo')
    ap.add_argument('--phi-far', type=float, default=0.25)
    ap.add_argument('--phi-near', type=float, default=0.65)
    ap.add_argument('--phi-radius', type=float, default=16.0)
    ap.add_argument('--w-win', type=float, default=1.0)
    ap.add_argument('--lam-drop', type=float, default=0.25)
    ap.add_argument('--p-self', type=float, default=0.5)
    ap.add_argument('--max-grad', type=float, default=1.0)
    ap.add_argument('--target-kl', type=float, default=0.03)
    ap.add_argument('--snap-every', type=int, default=10)
    ap.add_argument('--spawn-warmup', type=int, default=40,
                    help='training-time exploration: first N iters spawn at a fixed rate, spawn head untrained')
    ap.add_argument('--spawn-explore', type=float, default=0.3)
    ap.add_argument('--curr-iters', type=int, default=150, help='curriculum: larger random starting banks, annealed to 5000')
    ap.add_argument('--curr-max-bank', type=int, default=30000)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--out', default=os.path.join(HERE, 'checkpoints'))
    ap.add_argument('--resume', default=None)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    start_it = 0
    if args.resume:
        net, extra = load(args.resume)
        start_it = extra.get('iter', 0)
    else:
        net = Net(args.ch, args.blocks)
    net = net.to(memory_format=torch.channels_last)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    if args.resume and os.path.exists(args.resume + '.opt'):
        opt.load_state_dict(torch.load(args.resume + '.opt'))
    league = League(args.out)
    logf = open(os.path.join(args.out, 'train_log.jsonl'), 'a')
    for it in range(start_it + 1, start_it + args.iters + 1):
        t0 = time.time()
        args.it = it
        net.eval()
        data, scores, stats = rollout(net, league, args, rng)
        t1 = time.time()
        net.train()
        info = ppo_update(net, opt, data, args)
        t2 = time.time()
        mine = np.array([s[0] for s in scores])
        theirs = np.array([s[1] for s in scores])
        vs_snap = [s for s in scores if s[2] != 'L']
        rec = {
            'iter': it, 'games': args.envs, 'mean_bank': float(mine.mean()),
            'win_snap': float(np.mean([s[0] > s[1] for s in vs_snap])) if vs_snap else None,
            'n_snap': len(vs_snap), 'obs': len(data['obs']), 'ship_samples': len(data['sh_act']),
            'spawn_rate': float(data['sp_act'].mean()) if len(data['sp_act']) else 0.0,
            't_roll': round(t1 - t0, 1), 't_upd': round(t2 - t1, 1), **stats, **info,
        }
        logf.write(json.dumps(rec) + '\n')
        logf.flush()
        print(f"it {it} dep {stats['deposited']:.0f} bank {rec['mean_bank']:.0f} win_snap {rec['win_snap']} spawns {stats['spawns']} "
              f"drops {stats['drops']} kills {stats['kills']} ent {info.get('ent', 0):.3f} kl {info.get('kl', 0):.4f} "
              f"vl {info.get('vl', 0):.3f} sp_rate {rec['spawn_rate']:.2f} t {rec['t_roll']}+{rec['t_upd']}s", flush=True)
        latest = os.path.join(args.out, 'latest.pt')
        save(net, latest, {'iter': it})
        torch.save(opt.state_dict(), latest + '.opt')
        if it % args.snap_every == 0:
            save(net, os.path.join(args.out, f'snap_{it:05d}.pt'), {'iter': it})


if __name__ == '__main__':
    main()
