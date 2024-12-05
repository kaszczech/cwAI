import os
os.environ['JAX_ENABLE_X64'] = 'True'

import argparse
from collections import deque

from tqdm import tqdm
import jax
import numpy as np
from py_interface import *
from reinforced_lib import RLib
from reinforced_lib.agents.mab import *
from reinforced_lib.exts import BasicMab
from reinforced_lib.logs import *

from mldr.envs.ns3_ai_structures import Env, Act


MEMBLOCK_KEY = 2333
MEM_SIZE = 128

N_CW = 7
N_RTS_CTS = 2
N_AMPDU = 2

ACTION_HISTORY_LEN = 20
ACTION_PROB_THRESHOLD = 0.9
LATENCY_THRESHOLD = 0.01

AGENT_ARGS = {
    'EGreedy': {
        'e': 0.05,
        'optimistic_start': 1.0
    },
    'UCB': {
        'c': 0.01
    },
    'NormalThompsonSampling': {
        'alpha': 1.0,
        'beta': 1.0,
        'mu': 1.0,
        'lam': 0.0,
    }
}


if __name__ == '__main__':
    array_thr = [100]
    agents_list = ["UCB"]
    for agent_name in agents_list:
        for thr in tqdm(array_thr):

            logs_name = f"logs_szescn_n5_{agent_name}_{thr}.csv"
            csvPath_name = f"testy_szescn_n5_{agent_name}_{thr}.csv"

            args = argparse.ArgumentParser()

            # global settings
            args.add_argument('--seed', type=int, default=1001)
            args.add_argument('--mempoolKey', type=int, default=1234)
            args.add_argument('--ns3Path', type=str, default='')
            args.add_argument('--scenario', type=str, default='scenario_mgr')

            # ns-3 args
            args.add_argument('--agentName', type=str, default=agent_name)
            args.add_argument('--agentNumber', type=int, default=1)
            args.add_argument('--ampdu', action=argparse.BooleanOptionalAction, default=True)
            args.add_argument('--channelWidth', type=int, default=20)
            args.add_argument('--csvLogPath', type=str, default=logs_name)
            args.add_argument('--csvPath', type=str, default=csvPath_name)
            args.add_argument('--cw', type=int, default=-1)
            args.add_argument('--dataRate', type=int, default=thr) # TOSIE ZMIENIA
            args.add_argument('--distance', type=float, default=10.0)
            args.add_argument('--flowmonPath', type=str, default='flowmon.xml')
            args.add_argument('--fuzzTime', type=float, default=5.0)
            args.add_argument('--interactionTime', type=float, default=0.5)
            args.add_argument('--interPacketInterval', type=float, default=0.5)
            args.add_argument('--maxQueueSize', type=int, default=100)
            args.add_argument('--mcs', type=int, default=0)
            args.add_argument('--nWifi', type=int, default=5)
            args.add_argument('--packetSize', type=int, default=1500)
            args.add_argument('--rtsCts', action=argparse.BooleanOptionalAction, default=False)
            args.add_argument('--simulationTime', type=float, default=50.0)
            args.add_argument('--thrPath', type=str, default='thr.txt')

            # reward weights
            args.add_argument('--massive', type=float, default=0.0)
            args.add_argument('--throughput', type=float, default=1.0)
            args.add_argument('--urllc', type=float, default=0.0)

            # agent settings
            args.add_argument('--maxWarmup', type=int, default=50.0)
            args.add_argument('--useWarmup', action=argparse.BooleanOptionalAction, default=False)

            args = args.parse_args()
            args = vars(args)

            # read the arguments
            ns3_path = args.pop('ns3Path')

            if args['scenario'] == 'scenario_mgr':
                del args['interPacketInterval']
                del args['mcs']
                del args['thrPath']
                dataRate = min(115, args['dataRate'] * args['nWifi'])
            elif args['scenario'] == 'adhoc':
                del args['dataRate']
                del args['maxQueueSize']
                dataRate = (args['packetSize'] * args['nWifi'] / args['interPacketInterval']) / 1e6

            ns3_path = "/home/student/magisterka/ns-allinone-3.42/ns-3.42"

            seed = args.pop('seed')
            key = jax.random.PRNGKey(seed)

            agent = args['agentName']
            mempool_key = args.pop('mempoolKey')
            scenario = args.pop('scenario')

            ns3_args = args
            ns3_args['RngRun'] = seed

            # set up the reward function
            reward_probs = np.asarray([args.pop('massive'), args.pop('throughput'), args.pop('urllc')])

            def normalize_rewards(env):
                fairness = 1 + 10 * (env.fairness - 1)
                throughput = env.throughput / dataRate
                latency = 1 - env.latency / LATENCY_THRESHOLD

                rewards = np.asarray([fairness, throughput, latency])
                return np.dot(reward_probs, rewards)

            # set up the warmup function
            max_warmup = args.pop('maxWarmup')
            use_warmup = args.pop('useWarmup')

            action_history = {
                'cw': deque(maxlen=ACTION_HISTORY_LEN),
            }

            def end_warmup(cw, time):
                if not use_warmup or time > max_warmup:
                    return True

                action_history['cw'].append(cw)

                if len(action_history['cw']) < ACTION_HISTORY_LEN:
                    return False

                max_prob = lambda actions: (np.unique(actions, return_counts=True)[1] / len(actions)).max()

                if min(max_prob(action_history['cw'])) > ACTION_PROB_THRESHOLD:
                    return True

                return False

            # set up the agent
            if agent == 'wifi':
                rlib = None
            elif agent not in AGENT_ARGS:
                raise ValueError('Invalid agent type')
            else:
                rlib = RLib(
                    agent_type=globals()[agent],
                    agent_params=AGENT_ARGS[agent],
                    ext_type=BasicMab,
                    ext_params={'n_arms': N_CW},
                    logger_types=CsvLogger,
                    logger_params={'csv_path': f'rlib_{args["csvPath"]}'},
                    logger_sources=('reward', SourceType.METRIC)
                )
                rlib.init(seed)

            # set up the environment
            exp = Experiment(mempool_key, MEM_SIZE, scenario, ns3_path, using_waf=False)
            var = Ns3AIRL(MEMBLOCK_KEY, Env, Act)

            try:
                # run the experiment
                ns3_process = exp.run(setting=ns3_args, show_output=True)

                while not var.isFinish():
                    with var as data:
                        if data is None:
                            break

                        key, subkey = jax.random.split(key)
                        reward = normalize_rewards(data.env)

                        action = rlib.sample(reward)
                        cw, rts_cts, ampdu = np.unravel_index(action, (N_CW, 1, 2))

                        rlib.log('cw', cw)

                        data.act.cw = cw
                        data.act.end_warmup = end_warmup(cw, data.env.time)

                ns3_process.wait()
            finally:
                del exp
                del rlib
