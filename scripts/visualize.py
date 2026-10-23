"""
Visualization module for Athena (Oogway) contention experiments.

Parses ChampSim .out files directly from $ATHENA_HOME/experiments/<exp>/ and
writes a set of figures into $ATHENA_HOME/experiments/results/<exp>/:

    01_ipc.png               Victim/attacker IPC for every run (one bar plot),
                             plus victim IPC normalized to the Baseline run
                             (the "did Athena recover the victim?" panel).
    02_qtable_<run>.png      Final Oogway Q-cube for each Athena run, one
                             column per core: Q(s,a) heatmap (summed over
                             hash planes), greedy-action map with ties marked,
                             and the Q distribution per action.
    03_athena_internals.png  Action histogram, per-action reward components
                             (CYCLE highlighted = only weighted term), and
                             decision quality (ties, Q coverage, throttling).
    04_memory_system.png     LLC / prefetcher / OCP / DRAM stats that feed
                             Athena's state and reward, per run and core.
    summary.csv              Every derived metric, one row per (run, core).

Entry points:
    visualize_results(cd, csv_path=None, output=None)
        One experiment configuration (cd must be in config.EXPERIMENTS). This is
        what scripts/athena.py calls for `python athena.py -V <cd>`.
    visualize_all(cds=None, overview=True)
        Several experiments at once + the cross-experiment overview_ipc.png;
        cds defaults to every experiment directory found on disk.

Usage:
    python athena.py -V <cd>             # via the Athena CLI
    python visualize.py <cd>             # one experiment directly
    python visualize.py --all            # every experiment + overview
    python visualize.py                  # same as --all

The victim core is worked out per run from the `CPU_N runs <trace>` lines, not
assumed to be core 1 (some experiments, e.g. *_L1, put libquantum on core 0).
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

from config import EXPERIMENTS, DEFAULT_RESULTS_DIR, get_athena_home as _cfg_athena_home

# ---------------------------------------------------------------------------
# Style (validated reference palette: categorical slots in fixed order,
# blue<->red diverging with a gray midpoint)
# ---------------------------------------------------------------------------
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
GRID = '#e4e3df'
NEUTRAL = '#f0efec'
UNVISITED = '#d9d8d3'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100',
          '#e87ba4', '#008300', '#6250d6', '#e34948']
ROLE_COLOR = {'victim': SERIES[0], 'attacker': SERIES[1]}
ACTION_CMAP = ListedColormap(SERIES[:4])
Q_CMAP = LinearSegmentedColormap.from_list('q_div', ['#e34948', NEUTRAL, '#2a78d6'])

plt.rcParams.update({
    'figure.facecolor': SURFACE,
    'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE,
    'axes.edgecolor': GRID,
    'axes.labelcolor': INK_2,
    'axes.titlecolor': INK,
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'axes.grid': True,
    'axes.axisbelow': True,
    'grid.color': GRID,
    'grid.linewidth': 0.6,
    'xtick.color': INK_2,
    'ytick.color': INK_2,
    'text.color': INK,
    'legend.frameon': False,
    'font.size': 9,
})

# Oogway reward components, in the order of og_reward_exponent / action_rewards
REWARD_METRICS = ['MISPRED_BRANCH', 'LOAD_INST_ISSUE', 'L1D_LOAD_MISS',
                  'CYCLE', 'LLC_LOAD_MISS', 'LLC_MISS_LAT']
# Only CYCLE carries weight in og_reward_exponent (exponent 1.6); highlight it.
REWARD_WEIGHTED = {'CYCLE'}

# CD1 action semantics (4 actions)
ACTION_LABELS = ['OCP only', 'neither', 'pref only', 'both']

RUN_ORDER = ['Baseline', 'Naive', 'Athena']
RUN_CLASS = [('Baseline', 'Baseline'), ('Naive', 'Naive'), ('Athena', 'Athena')]

NUM = r'-?(?:\d+\.?\d*(?:e[+-]?\d+)?|nan|inf)'
STAT_RE = re.compile(rf'^([A-Za-z][\w.]*)\s+({NUM})\s*$', re.IGNORECASE)
CORE_STAT_RE = re.compile(rf'^Core_(\d+)_(\S+)\s+({NUM})\s*$', re.IGNORECASE)
REWARD_RE = re.compile(rf'^Core_(\d+)_og_action_(\d+)_reward\s+((?:{NUM}\s*)+)$',
                       re.IGNORECASE)
EPOCH_RE = re.compile(r'^Core_(\d+)_og_nr_epochs\b')
DIM_RE = re.compile(r'^Dim (\d+):(.*)$')
CPU_RUN_RE = re.compile(r'^CPU_(\d+) runs\s+(\S+)')
TRACES_RE = re.compile(r'-traces\s+(.*)$')


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def get_athena_home() -> str:
    """$ATHENA_HOME (via config.py), falling back to the repo root."""
    try:
        return _cfg_athena_home()
    except Exception:
        return os.environ.get('ATHENA_HOME') or str(Path(__file__).resolve().parents[1])


def get_results_dir() -> str:
    """$ATHENA_HOME/experiments/results (DEFAULT_RESULTS_DIR from config.py)."""
    return os.path.join(get_athena_home(), DEFAULT_RESULTS_DIR)


def get_experiments_dir() -> str:
    return os.path.join(get_athena_home(), 'experiments')


def list_experiments() -> List[str]:
    root = get_experiments_dir()
    return sorted(d for d in os.listdir(root)
                  if d != 'results' and os.path.isdir(os.path.join(root, d))
                  and any(f.endswith('.out') for f in os.listdir(os.path.join(root, d))))


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def _short_trace(path: str) -> str:
    name = os.path.basename(path)
    for ext in ('.champsimtrace.xz', '.champsimtrace.gz', '.champsimtrace',
                '.trace.xz', '.trace.gz', '.xz', '.gz', '.trace'):
        if name.endswith(ext):
            name = name[:-len(ext)]
            break
    return name


def _f(x: str) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float('nan')


def _classify_run(fname: str) -> Optional[str]:
    """Map a .out filename to Baseline / Naive / Athena, or None."""
    for key, label in RUN_CLASS:
        if key.lower() in fname.lower():
            return label
    return None


class RunData:
    """Everything parsed from a single .out file."""

    def __init__(self, label: str, path: str):
        self.label = label
        self.path = path
        self.exp = os.path.basename(os.path.dirname(path))
        # per-core
        self.trace: Dict[int, str] = {}
        self.role: Dict[int, str] = {}            # core -> 'victim'/'attacker'
        self.stats: Dict[int, Dict[str, float]] = {}   # core -> {stat: val}
        self.rewards: Dict[int, Dict[int, List[float]]] = {}  # core -> action -> [6]
        self.qcube: Dict[int, np.ndarray] = {}    # core -> (64, planes, 4)
        # global / shared
        self.g: Dict[str, float] = {}
        self.cmd_trace_order: List[str] = []

    def core_stat(self, core: int, name: str, default=float('nan')) -> float:
        return self.stats.get(core, {}).get(name, default)

    @property
    def is_athena(self) -> bool:
        return self.label == 'Athena'

    def victim_core(self) -> Optional[int]:
        for c, r in self.role.items():
            if r == 'victim':
                return c
        return None

    def attacker_core(self) -> Optional[int]:
        for c, r in self.role.items():
            if r == 'attacker':
                return c
        return None


def _assign_roles(run: RunData) -> None:
    """Victim = the real workload (libquantum/SPEC); attacker = the exp trace."""
    cores = sorted(run.trace)
    if not cores:
        return
    exp = run.exp.lower()

    def is_attacker(name: str) -> bool:
        n = name.lower()
        return (n == exp or n in exp or exp in n
                or n.startswith('mosaic') or 'charybdis' in n
                or 'destroyer' in n or 'contention' in n or 'cd1' in n)

    def is_victim(name: str) -> bool:
        n = name.lower()
        return 'libquantum' in n or bool(re.match(r'\d{3}\.', n))

    attacker = [c for c in cores if is_attacker(run.trace[c]) and not is_victim(run.trace[c])]
    victim = [c for c in cores if is_victim(run.trace[c])]

    if len(victim) == 1:
        v = victim[0]
    elif len(attacker) == 1 and len(cores) == 2:
        v = [c for c in cores if c != attacker[0]][0]
    else:
        # fallback: assume higher-IPC real workload is on the core NOT matching exp
        v = cores[-1]

    for c in cores:
        run.role[c] = 'victim' if c == v else 'attacker'


def parse_out(path: str) -> Optional[RunData]:
    label = _classify_run(os.path.basename(path))
    if label is None:
        return None
    run = RunData(label, path)

    # pending Q-dump accumulation
    dump_rows: List[np.ndarray] = []
    last_epoch_core: Optional[int] = None

    def flush_dump():
        nonlocal dump_rows
        if len(dump_rows) == 64 and last_epoch_core is not None:
            # keep the most recent completed dump for this core (final table)
            run.qcube[last_epoch_core] = np.stack(dump_rows, axis=0)  # (64, planes, 4)
        dump_rows = []

    with open(path, 'r', errors='replace') as fh:
        for line in fh:
            line = line.rstrip('\n')

            # trace / core mapping
            m = CPU_RUN_RE.match(line)
            if m:
                run.trace[int(m.group(1))] = _short_trace(m.group(2))
                continue
            if line.startswith('Original cmd'):
                mt = TRACES_RE.search(line)
                if mt:
                    run.cmd_trace_order = [_short_trace(t) for t in mt.group(1).split()]
                continue

            # Q-cube dump lines
            md = DIM_RE.match(line)
            if md:
                idx = int(md.group(1))
                if idx == 0 and dump_rows:
                    flush_dump()
                planes = [p for p in md.group(2).split('|') if p.strip()]
                vals = [[_f(x) for x in p.split()] for p in planes]
                # normalize ragged rows to common width
                width = max((len(v) for v in vals), default=0)
                if width:
                    arr = np.full((len(vals), width), np.nan)
                    for i, v in enumerate(vals):
                        arr[i, :len(v)] = v
                    dump_rows.append(arr)
                continue
            elif dump_rows:
                flush_dump()

            # per-core epoch marker (anchors Q-dump + leh_action_* attribution)
            me = EPOCH_RE.match(line)
            if me:
                last_epoch_core = int(me.group(1))

            # per-action reward vectors
            mr = REWARD_RE.match(line)
            if mr:
                core, action = int(mr.group(1)), int(mr.group(2))
                vec = [_f(x) for x in mr.group(3).split()]
                if not all(np.isnan(vec)):
                    run.rewards.setdefault(core, {})[action] = vec
                continue

            # Core_N_<stat> numeric
            mc = CORE_STAT_RE.match(line)
            if mc:
                core, name, val = int(mc.group(1)), mc.group(2), _f(mc.group(3))
                run.stats.setdefault(core, {})[name] = val
                continue

            # unprefixed numeric stat (global, or per-core via last_epoch_core)
            ms = STAT_RE.match(line)
            if ms:
                name, val = ms.group(1), _f(ms.group(2))
                if name.startswith('leh_action') and last_epoch_core is not None:
                    run.stats.setdefault(last_epoch_core, {})[name] = val
                else:
                    # first occurrence wins (scooby/Channel dup identically per core)
                    run.g.setdefault(name, val)
                continue

        flush_dump()

    _assign_roles(run)
    return run


def parse_experiment(exp: str) -> Dict[str, RunData]:
    d = os.path.join(get_experiments_dir(), exp)
    runs: Dict[str, RunData] = {}
    for f in sorted(os.listdir(d)):
        if not f.endswith('.out'):
            continue
        r = parse_out(os.path.join(d, f))
        if r:
            runs[r.label] = r
    return {k: runs[k] for k in RUN_ORDER if k in runs}


# ---------------------------------------------------------------------------
# Helpers for Q-cube
# ---------------------------------------------------------------------------
def q_summed(cube: np.ndarray) -> np.ndarray:
    """(buckets, planes, 4) -> Q summed over hash planes = (buckets, 4).

    NB the row axis is the LEH table index (leh_plane_dim=64 hash buckets),
    not a semantic state feature: a real state hashes to one bucket per plane.
    """
    return np.nansum(cube, axis=1)


def greedy_actions(q2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (argmax action per hash bucket, tie-mask per hash bucket)."""
    best = np.nanmax(q2d, axis=1, keepdims=True)
    ismax = np.isclose(q2d, best)
    arg = np.nanargmax(q2d, axis=1)
    tie = ismax.sum(axis=1) > 1
    return arg, tie


# ---------------------------------------------------------------------------
# Figure 1: IPC
# ---------------------------------------------------------------------------
def fig_ipc(runs: Dict[str, RunData], out: str, exp: str) -> None:
    labels = [l for l in RUN_ORDER if l in runs]
    if not labels:
        return

    fig = plt.figure(figsize=(11, 4.6))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.35, 1.0], wspace=0.26)
    ax0, ax1 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    x = np.arange(len(labels))
    w = 0.38
    vic, att = [], []
    for l in labels:
        r = runs[l]
        vc, ac = r.victim_core(), r.attacker_core()
        vic.append(r.core_stat(vc, 'cumulative_IPC') if vc is not None else np.nan)
        att.append(r.core_stat(ac, 'cumulative_IPC') if ac is not None else np.nan)

    b0 = ax0.bar(x - w / 2, vic, w, color=ROLE_COLOR['victim'], label='victim (workload)')
    b1 = ax0.bar(x + w / 2, att, w, color=ROLE_COLOR['attacker'], label='attacker (core-0 code)')
    for bars in (b0, b1):
        ax0.bar_label(bars, fmt='%.3f', padding=2, fontsize=7.5, color=INK_2)
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels)
    ax0.set_ylabel('cumulative IPC')
    ax0.set_title('IPC per run')
    ax0.legend(loc='upper left', fontsize=8)
    ax0.margins(y=0.16)

    # victim IPC normalized to Baseline victim
    base = vic[labels.index('Baseline')] if 'Baseline' in labels else np.nan
    norm = [v / base if base else np.nan for v in vic]
    colors = [SERIES[2] if (n is not None and n >= 1) else SERIES[7] for n in norm]
    bb = ax1.bar(x, norm, 0.6, color=colors)
    ax1.bar_label(bb, fmt='%.2fx', padding=2, fontsize=8, color=INK_2)
    ax1.axhline(1.0, color=INK, lw=1.0, ls='--')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel('victim IPC / Baseline victim IPC')
    ax1.set_title('Victim recovery\n(>1 = Athena healed the victim = attack lost)')
    ax1.margins(y=0.18)

    fig.suptitle(f'{exp}  —  core IPC', fontsize=12, fontweight='bold', y=1.02)
    fig.savefig(out, bbox_inches='tight', dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Q-table (one per Athena run)
# ---------------------------------------------------------------------------
def fig_qtable(run: RunData, out: str) -> None:
    cores = [c for c in sorted(run.qcube)]
    if not cores:
        return
    ncol = len(cores)
    fig = plt.figure(figsize=(5.4 * ncol, 8.2))
    gs = GridSpec(3, ncol, figure=fig, height_ratios=[3.0, 0.5, 1.2],
                  hspace=0.42, wspace=0.32)

    for j, core in enumerate(cores):
        cube = run.qcube[core]
        q2d = q_summed(cube)                       # (64, 4)
        role = run.role.get(core, f'core{core}')
        trace = run.trace.get(core, '')
        arg, tie = greedy_actions(q2d)

        # --- Q heatmap: dims (y) x actions (x) ---
        axh = fig.add_subplot(gs[0, j])
        vmax = np.nanmax(np.abs(q2d)) or 1.0
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        im = axh.imshow(q2d, aspect='auto', cmap=Q_CMAP, norm=norm,
                        interpolation='nearest')
        axh.set_title(f'core {core} · {role}\n{trace}', fontsize=9.5)
        axh.set_xlabel('action')
        axh.set_ylabel('hash bucket (0-63)')
        axh.set_xticks(range(q2d.shape[1]))
        axh.set_xticklabels([f'{i}\n{ACTION_LABELS[i]}' if i < len(ACTION_LABELS) else str(i)
                             for i in range(q2d.shape[1])], fontsize=7)
        axh.grid(False)
        cb = fig.colorbar(im, ax=axh, fraction=0.046, pad=0.03)
        cb.set_label('Q(s,a) summed over planes', fontsize=7.5)
        cb.ax.tick_params(labelsize=7)

        # --- greedy-action strip ---
        axg = fig.add_subplot(gs[1, j])
        strip = arg.reshape(1, -1).astype(float)
        axg.imshow(strip, aspect='auto', cmap=ACTION_CMAP, vmin=0, vmax=3,
                   interpolation='nearest')
        # mark ties (unlearned states) with a dot
        tie_x = np.where(tie)[0]
        if tie_x.size:
            axg.scatter(tie_x, np.zeros_like(tie_x), s=10, marker='o',
                        facecolors='none', edgecolors=INK, linewidths=0.7)
        axg.set_yticks([])
        axg.set_xlabel('hash bucket  →  greedy action (○ = Q tie / unlearned)')
        axg.set_title(f'greedy policy   ties {int(tie.sum())}/64', fontsize=8.5)
        axg.grid(False)

        # --- Q distribution per action ---
        axd = fig.add_subplot(gs[2, j])
        data = [q2d[:, a][~np.isnan(q2d[:, a])] for a in range(q2d.shape[1])]
        bp = axd.boxplot(data, patch_artist=True, widths=0.6,
                         medianprops=dict(color=INK, lw=1.2),
                         flierprops=dict(marker='.', markersize=3,
                                         markeredgecolor=INK_2))
        for patch, a in zip(bp['boxes'], range(len(data))):
            patch.set_facecolor(SERIES[a % 4])
            patch.set_alpha(0.75)
            patch.set_edgecolor(INK_2)
        axd.axhline(0, color=INK_2, lw=0.8, ls=':')
        axd.set_xticks(range(1, len(data) + 1))
        axd.set_xticklabels([ACTION_LABELS[a] if a < len(ACTION_LABELS) else str(a)
                             for a in range(len(data))], fontsize=7.5, rotation=20)
        axd.set_ylabel('Q per bucket')
        axd.set_title('Q distribution by action', fontsize=8.5)

    # action legend
    handles = [Patch(facecolor=SERIES[a], label=f'{a}: {ACTION_LABELS[a]}')
               for a in range(4)]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f'{run.exp}  —  Athena Q-cube (final)', fontsize=12,
                 fontweight='bold', y=0.995)
    fig.savefig(out, bbox_inches='tight', dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Athena internals
# ---------------------------------------------------------------------------
def fig_internals(run: RunData, out: str) -> None:
    cores = sorted(run.rewards) or sorted(run.stats)
    # restrict to cores that actually have og action data
    acores = [c for c in sorted(run.stats)
              if any(k.startswith('og_action_') and k.count('_') == 2
                     for k in run.stats[c])]
    if not acores:
        acores = sorted(run.rewards)
    if not acores:
        return

    fig = plt.figure(figsize=(13, 4.6 * len(acores)))
    gs = GridSpec(len(acores), 3, figure=fig, width_ratios=[1.0, 1.5, 1.0],
                  hspace=0.5, wspace=0.34)

    for i, core in enumerate(acores):
        role = run.role.get(core, f'core{core}')
        st = run.stats.get(core, {})

        # --- action histogram ---
        axh = fig.add_subplot(gs[i, 0])
        counts = [st.get(f'og_action_{a}', 0.0) for a in range(4)]
        bars = axh.bar(range(4), counts, color=[SERIES[a] for a in range(4)])
        axh.bar_label(bars, fmt='%.0f', fontsize=7, color=INK_2, padding=2)
        axh.set_xticks(range(4))
        axh.set_xticklabels([f'{a}\n{ACTION_LABELS[a]}' for a in range(4)], fontsize=7.5)
        axh.set_ylabel('times chosen')
        axh.set_title(f'core {core} · {role}\naction histogram', fontsize=9.5)
        axh.margins(y=0.14)

        # --- per-action reward components ---
        axr = fig.add_subplot(gs[i, 1])
        rw = run.rewards.get(core, {})
        na = max([4] + list(rw))
        xm = np.arange(len(REWARD_METRICS))
        bw = 0.8 / max(1, len([a for a in range(4) if a in rw]))
        k = 0
        for a in range(4):
            if a not in rw:
                continue
            vals = rw[a][:len(REWARD_METRICS)]
            axr.bar(xm + k * bw, vals, bw, color=SERIES[a],
                    label=f'{a}:{ACTION_LABELS[a]}')
            k += 1
        axr.axhline(0, color=INK_2, lw=0.8)
        axr.set_xticks(xm + (k - 1) * bw / 2)
        axr.set_xticklabels(REWARD_METRICS, fontsize=7, rotation=25, ha='right')
        # highlight the only weighted term
        for lab in axr.get_xticklabels():
            if lab.get_text() in REWARD_WEIGHTED:
                lab.set_color(SERIES[7])
                lab.set_fontweight('bold')
        axr.set_ylabel('mean per-epoch reward')
        axr.set_title('reward components by action\n(CYCLE = only weighted term)',
                      fontsize=9.5)
        axr.legend(fontsize=7, ncol=2)

        # --- decision quality ---
        axq = fig.add_subplot(gs[i, 2])
        called = st.get('leh_action_called', np.nan)
        ties = st.get('leh_action_tie', np.nan)
        tie_rate = (ties / called * 100) if called else np.nan
        cov = np.nan
        if core in run.qcube:
            q2d = q_summed(run.qcube[core])
            cov = np.mean(np.any(~np.isclose(np.nan_to_num(q2d), 0), axis=1)) * 100
        pf = st.get('og_pf_throttled', np.nan)
        ocp = st.get('og_ocp_throttled', np.nan)

        names = ['tie rate\n%', 'Q coverage\n%', 'pf throttled', 'ocp throttled']
        vals = [tie_rate, cov, pf, ocp]
        cols = [SERIES[7], SERIES[2], SERIES[3], SERIES[4]]
        # two scales: percentages left, counts right
        axq.bar([0, 1], [tie_rate, cov], color=[cols[0], cols[1]])
        axq.set_ylim(0, 100)
        axq.set_ylabel('%')
        axq2 = axq.twinx()
        axq2.bar([2, 3], [pf, ocp], color=[cols[2], cols[3]])
        axq2.set_ylabel('count')
        axq2.grid(False)
        axq.set_xticks(range(4))
        axq.set_xticklabels(names, fontsize=7.5)
        axq.set_title('decision quality + throttling', fontsize=9.5)
        for xi, v in enumerate(vals):
            if np.isnan(v):
                continue
            axt = axq if xi < 2 else axq2
            axt.text(xi, v, f'{v:.1f}' if xi < 2 else f'{v:.0f}',
                     ha='center', va='bottom', fontsize=7, color=INK_2)

    fig.suptitle(f'{run.exp}  —  Athena internals', fontsize=12,
                 fontweight='bold', y=1.0)
    fig.savefig(out, bbox_inches='tight', dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: Memory system
# ---------------------------------------------------------------------------
def fig_memory(runs: Dict[str, RunData], out: str, exp: str) -> None:
    labels = [l for l in RUN_ORDER if l in runs]
    if not labels:
        return
    x = np.arange(len(labels))

    fig = plt.figure(figsize=(13, 9))
    gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.34)

    def per_core_grouped(ax, stat, title, ylabel, scale=1.0):
        w = 0.38
        for off, role in ((-w / 2, 'victim'), (w / 2, 'attacker')):
            vals = []
            for l in labels:
                r = runs[l]
                c = r.victim_core() if role == 'victim' else r.attacker_core()
                vals.append(r.core_stat(c, stat) * scale if c is not None else np.nan)
            ax.bar(x + off, vals, w, color=ROLE_COLOR[role], label=role)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9.5)

    # offchip predictor precision / recall (victim)
    ax = fig.add_subplot(gs[0, 0])
    w = 0.38
    for off, stat, col, lab in ((-w / 2, 'offchip_pred_precision', SERIES[0], 'precision'),
                                (w / 2, 'offchip_pred_recall', SERIES[1], 'recall')):
        vals = [runs[l].core_stat(runs[l].victim_core(), stat) for l in labels]
        ax.bar(x + off, vals, w, color=col, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('%'); ax.set_title('victim off-chip predictor', fontsize=9.5)
    ax.legend(fontsize=7.5)

    ax = fig.add_subplot(gs[0, 1])
    per_core_grouped(ax, 'LLC_average_miss_latency', 'LLC avg miss latency',
                     'cycles')
    ax.legend(fontsize=7.5)

    ax = fig.add_subplot(gs[0, 2])
    per_core_grouped(ax, 'total_loads_went_offchip', 'loads gone off-chip',
                     'count')
    ax.legend(fontsize=7.5)

    # DRAM bandwidth occupancy (global scooby bw levels), stacked per run
    ax = fig.add_subplot(gs[1, 0])
    bw_cols = [UNVISITED, SERIES[3], SERIES[1], SERIES[7]]
    bottoms = np.zeros(len(labels))
    totals = []
    mats = []
    for l in labels:
        g = runs[l].g
        lv = np.array([g.get(f'scooby_bw_level_{k}', 0.0) for k in range(4)], float)
        mats.append(lv)
        totals.append(lv.sum())
    mats = np.array(mats)
    tot = mats.sum(axis=1, keepdims=True)
    frac = np.divide(mats, tot, out=np.zeros_like(mats), where=tot > 0) * 100
    for k in range(4):
        ax.bar(x, frac[:, k], 0.6, bottom=bottoms, color=bw_cols[k],
               label=f'bw level {k}')
        bottoms += frac[:, k]
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('% of epochs'); ax.set_title('DRAM bandwidth level (scooby)', fontsize=9.5)
    ax.legend(fontsize=7, ncol=2)

    # DRAM row-buffer hit rate + dbus congestion (global Channel_0)
    ax = fig.add_subplot(gs[1, 1])
    rq_hit = np.array([runs[l].g.get('Channel_0_RQ_row_buffer_hit', np.nan) for l in labels])
    rq_mis = np.array([runs[l].g.get('Channel_0_RQ_row_buffer_miss', np.nan) for l in labels])
    hitrate = rq_hit / (rq_hit + rq_mis) * 100
    bb = ax.bar(x, hitrate, 0.6, color=SERIES[0])
    ax.bar_label(bb, fmt='%.1f%%', fontsize=7.5, color=INK_2, padding=2)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('RQ row-buffer hit %'); ax.set_ylim(0, 100)
    ax.set_title('DRAM read row-buffer hit rate', fontsize=9.5)

    ax = fig.add_subplot(gs[1, 2])
    dbus = [runs[l].g.get('Channel_0_dbus_congested', np.nan) for l in labels]
    bb = ax.bar(x, dbus, 0.6, color=SERIES[7])
    ax.bar_label(bb, fmt='%.0f', fontsize=7, color=INK_2, padding=2)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('cycles'); ax.set_title('DRAM dbus congested', fontsize=9.5)
    ax.margins(y=0.14)

    fig.suptitle(f'{exp}  —  memory system (Athena state/reward inputs)',
                 fontsize=12, fontweight='bold', y=0.995)
    fig.savefig(out, bbox_inches='tight', dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# summary.csv
# ---------------------------------------------------------------------------
def write_summary(runs: Dict[str, RunData], out: str) -> pd.DataFrame:
    rows = []
    for label in RUN_ORDER:
        if label not in runs:
            continue
        r = runs[label]
        for core in sorted(r.trace):
            st = r.stats.get(core, {})
            q2d = q_summed(r.qcube[core]) if core in r.qcube else None
            cov = (np.mean(np.any(~np.isclose(np.nan_to_num(q2d), 0), axis=1)) * 100
                   if q2d is not None else np.nan)
            called = st.get('leh_action_called', np.nan)
            ties = st.get('leh_action_tie', np.nan)
            rows.append({
                'exp': r.exp,
                'run': label,
                'core': core,
                'role': r.role.get(core, ''),
                'trace': r.trace.get(core, ''),
                'IPC': st.get('cumulative_IPC', np.nan),
                'total_loads': st.get('total_loads', np.nan),
                'loads_offchip': st.get('total_loads_went_offchip', np.nan),
                'LLC_miss_lat': st.get('LLC_average_miss_latency', np.nan),
                'LLC_pref_useful': st.get('LLC_prefetch_useful', np.nan),
                'LLC_pref_late': st.get('LLC_prefetch_late', np.nan),
                'offchip_precision': st.get('offchip_pred_precision', np.nan),
                'offchip_recall': st.get('offchip_pred_recall', np.nan),
                'og_epochs': st.get('og_nr_epochs', np.nan),
                'og_action_0': st.get('og_action_0', np.nan),
                'og_action_1': st.get('og_action_1', np.nan),
                'og_action_2': st.get('og_action_2', np.nan),
                'og_action_3': st.get('og_action_3', np.nan),
                'og_pf_throttled': st.get('og_pf_throttled', np.nan),
                'og_ocp_throttled': st.get('og_ocp_throttled', np.nan),
                'leh_tie_rate_%': (ties / called * 100) if called else np.nan,
                'q_coverage_%': cov,
            })
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    return df


# ---------------------------------------------------------------------------
# Overview across experiments
# ---------------------------------------------------------------------------
def fig_overview(all_runs: Dict[str, Dict[str, RunData]], out: str) -> None:
    exps = sorted(all_runs)
    if not exps:
        return
    fig, ax = plt.subplots(figsize=(1.6 * len(exps) + 3, 5))
    x = np.arange(len(exps))
    w = 0.26
    for k, label in enumerate(RUN_ORDER):
        vals = []
        for e in exps:
            runs = all_runs[e]
            if label in runs:
                r = runs[label]
                vals.append(r.core_stat(r.victim_core(), 'cumulative_IPC'))
            else:
                vals.append(np.nan)
        ax.bar(x + (k - 1) * w, vals, w, color=SERIES[k], label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(exps, rotation=20, ha='right', fontsize=8)
    ax.set_ylabel('victim cumulative IPC')
    ax.set_title('Victim IPC across experiments (lower Athena vs Baseline = stronger attack)')
    ax.legend()
    fig.savefig(out, bbox_inches='tight', dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _render_experiment(runs: Dict[str, RunData], out_dir: str,
                       summary_csv: Optional[str] = None, exp: str = '') -> None:
    """Write the 01-04 figures + summary.csv for one already-parsed experiment."""
    os.makedirs(out_dir, exist_ok=True)

    fig_ipc(runs, os.path.join(out_dir, '01_ipc.png'), exp)
    for label, r in runs.items():
        if r.is_athena and r.qcube:
            fig_qtable(r, os.path.join(out_dir, f'02_qtable_{label}.png'))
        if r.is_athena:
            fig_internals(r, os.path.join(out_dir, '03_athena_internals.png'))
    fig_memory(runs, os.path.join(out_dir, '04_memory_system.png'), exp)
    write_summary(runs, summary_csv or os.path.join(out_dir, 'summary.csv'))


def _resolve_out_dir(cd: str, output: Optional[str]) -> str:
    """Default results/<cd>/; honor an explicit `output` dir or .png path."""
    if not output:
        return os.path.join(get_results_dir(), cd)
    # athena.py passes a single path; if it looks like a file, use its parent.
    return os.path.dirname(output) if output.lower().endswith('.png') else output


def visualize_results(cd: str, csv_path: Optional[str] = None,
                      output: Optional[str] = None) -> Dict[str, RunData]:
    """Visualize one experiment configuration (athena.py -V <cd> entry point).

    Parses the ChampSim .out files in $ATHENA_HOME/experiments/<cd>/ and writes
    a figure set into $ATHENA_HOME/experiments/results/<cd>/:
        01_ipc.png, 02_qtable_<run>.png, 03_athena_internals.png,
        04_memory_system.png, summary.csv.

    Args:
        cd: experiment key (must be in config.EXPERIMENTS).
        csv_path: optional path to write summary.csv (default: <out_dir>/summary.csv).
        output: optional output directory (or .png path, whose parent is used);
                default $ATHENA_HOME/experiments/results/<cd>/.

    Returns {run_label: RunData}.
    """
    if cd not in EXPERIMENTS:
        raise ValueError(f'Invalid experiment: {cd}. '
                         f'Must be one of: {list(EXPERIMENTS.keys())}')

    exp_dir = os.path.join(get_experiments_dir(), cd)
    if not os.path.isdir(exp_dir):
        raise FileNotFoundError(f'No experiment directory: {exp_dir}')

    print(f'Parsing .out files for {cd} ({EXPERIMENTS[cd].get("description", "")})...')
    runs = parse_experiment(cd)
    if not runs:
        print(f'No parseable runs in {exp_dir}')
        sys.exit(1)

    out_dir = _resolve_out_dir(cd, output)
    _render_experiment(runs, out_dir, summary_csv=csv_path, exp=cd)

    v = runs.get('Athena')
    vc = v.victim_core() if v else None
    print(f'[ok]   {cd}: {len(runs)} runs -> {out_dir}'
          + (f'  (victim=core{vc}, Athena IPC {v.core_stat(vc, "cumulative_IPC"):.4f})'
             if v and vc is not None else ''))
    return runs


def visualize_all(cds: Optional[List[str]] = None, overview: bool = True
                  ) -> Dict[str, Dict[str, RunData]]:
    """Render several experiments and (overview=True) the cross-experiment
    experiments/results/overview_ipc.png. cds defaults to every experiment
    found on disk under $ATHENA_HOME/experiments/."""
    if cds is None:
        cds = list_experiments()

    all_runs: Dict[str, Dict[str, RunData]] = {}
    for cd in cds:
        runs = parse_experiment(cd)
        if not runs:
            print(f'[skip] {cd}: no parseable runs')
            continue
        _render_experiment(runs, os.path.join(get_results_dir(), cd), exp=cd)
        all_runs[cd] = runs
        v = runs.get('Athena')
        vc = v.victim_core() if v else None
        print(f'[ok]   {cd}: {len(runs)} runs'
              + (f'  (victim=core{vc}, Athena IPC {v.core_stat(vc, "cumulative_IPC"):.4f})'
                 if v and vc is not None else ''))

    if overview and all_runs:
        os.makedirs(get_results_dir(), exist_ok=True)
        out = os.path.join(get_results_dir(), 'overview_ipc.png')
        fig_overview(all_runs, out)
        print(f'[ok]   overview -> {out}')
    return all_runs


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description='Visualize Athena contention experiments from ChampSim .out files')
    parser.add_argument('cd', nargs='?',
                        help='Experiment configuration (default: --all experiments)')
    parser.add_argument('--all', action='store_true',
                        help='Process every experiment on disk + overview_ipc.png')
    parser.add_argument('--csv', help='Path to write summary.csv (single-cd mode)')
    parser.add_argument('--output', '-o',
                        help='Output directory (or .png path) for the figures')
    args = parser.parse_args()

    if args.all or not args.cd:
        visualize_all()
    else:
        visualize_results(args.cd, csv_path=args.csv, output=args.output)
    return 0


if __name__ == '__main__':
    sys.exit(main())
