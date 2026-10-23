"""
Configuration file for Athena experiment management system.
Contains all constants, defaults, and configuration mappings.
"""

import os
from typing import Dict, List

# Default values for job generation
DEFAULT_NCORES = 1
DEFAULT_PARTITION = "debug"
DEFAULT_HOSTNAME = "agamarch"

# Experiment variable definitions
EXP_VARIABLES = {
    'BASE': '--warmup_instructions=10000000 --simulation_instructions=30000000 --llc_replacement_type=ship --config=$(ATHENA_HOME)/config/nopref.ini --rob_partition_size=64,128,320 --rob_frontal_partition_ids=0 --rob_dorsal_partition_ids=2 --dram_io_freq=400',

    # L1D prefetchers
    'IPCP': '--l1d_prefetcher_types=ipcp',
    'BERTI': '--l1d_prefetcher_types=berti',

    # L2C prefetchers
    'PYTHIA': '--l2c_prefetcher_types=scooby --config=$(ATHENA_HOME)/config/pythia.ini --l2c_prefetcher_force_prefetch_at_llc=true',
    'SMS': '--l2c_prefetcher_types=sms --config=$(ATHENA_HOME)/config/sms.ini --l2c_prefetcher_force_prefetch_at_llc=true',
    'SPP+PPF': '--l2c_prefetcher_types=spp_ppf_dev --config=$(ATHENA_HOME)/config/spp_ppf_dev.ini --l2c_prefetcher_force_prefetch_at_llc=true',
    'MLOP': '--l2c_prefetcher_types=mlop --config=$(ATHENA_HOME)/config/mlop.ini --l2c_prefetcher_force_prefetch_at_llc=true',

    # Off-Chip Predictors (OCP)
    'POPET': '--config=$(ATHENA_HOME)/config/hermes_base.ini --config=$(ATHENA_HOME)/config/ocp_hermes.ini',
    'HMP': '--config=$(ATHENA_HOME)/config/hermes_base.ini --config=$(ATHENA_HOME)/config/ocp_hmp_ensemble.ini',
    'TTP': '--config=$(ATHENA_HOME)/config/hermes_base.ini --config=$(ATHENA_HOME)/config/ocp_ttp.ini',

    # Coordination mechanisms
    'TLP': '--config=$(ATHENA_HOME)/config/hermes_base.ini --config=$(ATHENA_HOME)/config/ocp_tlp.ini',
    'HPAC': '--config=$(ATHENA_HOME)/config/oogway_dev.ini --heuristic_enable=true',
    'MAB': '--config=$(ATHENA_HOME)/config/oogway_dev.ini --mab_enable=true',
    'ATHENA': '--config=$(ATHENA_HOME)/config/oogway_dev.ini',

    # Cache Design configurations
    'CD1': '--og_multi_prefetcher_enable=false --og_prefetcher_level=2',  # OCP + L2C
    'CD2': '--og_multi_prefetcher_enable=false --og_prefetcher_level=1',  # OCP + L1D
    'CD3': '--leh_num_actions=8 --og_multi_prefetcher_enable=true --og_coordination_mode=l2c_l2c',  # OCP + 2 L2C
    'CD4': '--leh_num_actions=8 --og_multi_prefetcher_enable=true --og_coordination_mode=l1d_l2c',  # OCP + L1D + L2C
    'CD5': '--og_multi_prefetcher_enable=true --og_l2c_only_coordination=true',  # 2 L2C

    # OCP Request Latency configurations
    'LAT18': '--ddrp_req_latency=18',
    'LAT30': '--ddrp_req_latency=30',

    # Memory Bandwidth configurations
    'BW200': '--dram_io_freq=200',   # 1.6 GB/s
    'BW400': '--dram_io_freq=400',   # 3.2 GB/s (default in BASE)
    'BW800': '--dram_io_freq=800',   # 6.4 GB/s
    'BW1600': '--dram_io_freq=1600', # 12.8 GB/s
}

# Experiment configurations (merges experiments with their variable definitions)
EXPERIMENTS = {
    'mosaic_idle': {
        'description': 'Idle trace to measure baseline perf',
        'traces': ['idle__libquantum'],
        'experiments': {
            'Baseline': ['BASE'],
            'Naive-POPET-Pythia': ['BASE', 'POPET', 'PYTHIA'],
            'Athena-POPET-Pythia': ['BASE', 'POPET', 'PYTHIA', 'CD1', 'ATHENA'],
        }
    },
    'mosaic_cd1_charybdis_L1': {
        'description': 'Reward hack with CD1',
        'traces': ['mosaic_cd1_charybdis_L1__libquantum'],
        'experiments': {
            'Baseline': ['BASE'],
            'Naive-POPET-Pythia': ['BASE', 'POPET', 'PYTHIA'],
            'Athena-POPET-Pythia': ['BASE', 'POPET', 'PYTHIA', 'CD1', 'ATHENA'],
        }
    },
    'mosaic_cd1_charybdis_L2': {
        'description': 'Reward hack with CD1',
        'traces': ['mosaic_cd1_charybdis_L2__libquantum'],
        'experiments': {
            'Baseline': ['BASE'],
            'Naive-POPET-Pythia': ['BASE', 'POPET', 'PYTHIA'],
            'Athena-POPET-Pythia': ['BASE', 'POPET', 'PYTHIA', 'CD1', 'ATHENA'],
        }
    },
    'mosaic_cd1_charybdis_L345': {
        'description': 'Reward hack with CD1',
        'traces': ['mosaic_cd1_charybdis_L345__libquantum'],
        'experiments': {
            'Baseline': ['BASE'],
            'Naive-POPET-Pythia': ['BASE', 'POPET', 'PYTHIA'],
            'Athena-POPET-Pythia': ['BASE', 'POPET', 'PYTHIA', 'CD1', 'ATHENA'],
        }
    },
    'mosaic_cd1_plegma': {
        'description': 'Reward hack with CD1',
        'traces': ['mosaic_cd1_plegma__libquantum'],
        'experiments': {
            'Baseline': ['BASE'],
            'Naive-POPET-Pythia': ['BASE', 'POPET', 'PYTHIA'],
            'Athena-POPET-Pythia': ['BASE', 'POPET', 'PYTHIA', 'CD1', 'ATHENA'],
        }
    },
}


# Default paths
DEFAULT_RESULTS_DIR = 'experiments/results'  # Relative to ATHENA_HOME

# Get ATHENA_HOME from environment
ATHENA_HOME = os.environ.get('ATHENA_HOME', '')

def get_athena_home() -> str:
    """Get ATHENA_HOME from environment, raise error if not set."""
    if not ATHENA_HOME:
        raise ValueError("$ATHENA_HOME environment variable is not defined. Have you sourced setvars.sh?")
    return ATHENA_HOME

# Workload type enumeration
class WorkloadType:
    SPEC = 'SPEC'
    PARSEC = 'PARSEC'
    LIGRA = 'LIGRA'
    CVP = 'CVP'

# Trace data configuration (from hpca_final_1.tlist)
# Each entry contains: path, knobs, workload_type, adverse (prefetcher-adverse if True)
TRACE_DATA = {
    'idle__libquantum': {
        'path': '$ATHENA_HOME/traces/462.libquantum-1343B.champsimtrace.xz $ATHENA_HOME/traces/idle.champsimtrace.xz',
        'knobs': '',
        'type': WorkloadType.SPEC,
        'adverse': False,
    },
    'mosaic_cd1_llc_contention__libquantum': {
        'path': '$ATHENA_HOME/traces/mosaic_cd1_llc_contention.champsimtrace.xz $ATHENA_HOME/traces/462.libquantum-1343B.champsimtrace.xz',
        'knobs': '',
        'type': WorkloadType.SPEC,
        'adverse': False,
    },
    'mosaic_cd1_row_destroyer__libquantum': {
        'path': '$ATHENA_HOME/traces/mosaic_cd1_row_destroyer.champsimtrace.xz $ATHENA_HOME/traces/462.libquantum-1343B.champsimtrace.xz',
        'knobs': '',
        'type': WorkloadType.SPEC,
        'adverse': False,
    },
    'mosaic_cd1_charybdis_L1__libquantum': {
        'path': '$ATHENA_HOME/traces/462.libquantum-1343B.champsimtrace.xz $ATHENA_HOME/traces/mosaic_cd1_charybdis_L1.champsimtrace.xz',
        'knobs': '',
        'type': WorkloadType.SPEC,
        'adverse': False,
    },
   'mosaic_cd1_charybdis_L2__libquantum': {
        'path': '$ATHENA_HOME/traces/462.libquantum-1343B.champsimtrace.xz $ATHENA_HOME/traces/mosaic_cd1_charybdis_L2.champsimtrace.xz',
        'knobs': '',
        'type': WorkloadType.SPEC,
        'adverse': False,
    },
    'mosaic_cd1_charybdis_L345__libquantum': {
        'path': '$ATHENA_HOME/traces/462.libquantum-1343B.champsimtrace.xz $ATHENA_HOME/traces/mosaic_cd1_charybdis_L345.champsimtrace.xz',
        'knobs': '',
        'type': WorkloadType.SPEC,
        'adverse': False,
    },
    'mosaic_cd1_plegma__libquantum': {
        'path': '$ATHENA_HOME/traces/462.libquantum-1343B.champsimtrace.xz $ATHENA_HOME/traces/mosaic_cd1_plegma.champsimtrace.xz',
        'knobs': '',
        'type': WorkloadType.SPEC,
        'adverse': False,
    },
}

def get_trace_info(name: str) -> dict:
    """Get trace information including path, knobs, type, and adverse."""
    if name not in TRACE_DATA:
        raise ValueError(f"Unknown trace: {name}")
    return TRACE_DATA[name].copy()


def get_traces_by_type(workload_type: str) -> List[str]:
    """Get list of trace names filtered by workload type."""
    return [name for name, data in TRACE_DATA.items() if data['type'] == workload_type]


def get_all_trace_names() -> List[str]:
    """Get list of all trace names."""
    return list(TRACE_DATA.keys())


def get_cd_traces(cd: str) -> List[str]:
    """Get trace names a CD should run on.

    Uses EXPERIMENTS[cd]['traces'] if present; otherwise all traces.
    """
    if cd not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {cd}. Must be one of: {list(EXPERIMENTS.keys())}")
    names = EXPERIMENTS[cd].get('traces')
    if names is None:
        return get_all_trace_names()
    for n in names:
        if n not in TRACE_DATA:
            raise ValueError(f"Unknown trace '{n}' in EXPERIMENTS['{cd}']['traces']")
    return list(names)


# Default metrics configuration for rollup
DEFAULT_METRICS = [
    {'NAME': 'Core_0_cumulative_IPC', 'TYPE': 'sum'},
    {'NAME': 'Core_1_cumulative_IPC', 'TYPE': 'sum'},
]

# Extended metrics including coverage and accuracy
EXTENDED_METRICS = [
    {'NAME': 'Core_0_cumulative_IPC', 'TYPE': 'sum'},
    {'NAME': 'Core_0_offchip_pred_precision', 'TYPE': 'sum'},
    {'NAME': 'Core_0_offchip_pred_recall', 'TYPE': 'sum'},
]
