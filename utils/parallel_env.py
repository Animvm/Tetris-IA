import multiprocessing as mp
from multiprocessing import Process, Pipe
import numpy as np

# proceso worker para ejecutar ambiente en paralelo
def worker(remote, parent_remote, env_fn):
    parent_remote.close()
    env = env_fn()

    while True:
        try:
            cmd, data = remote.recv()

            if cmd == 'step':
                obs, reward, terminated, truncated, info = env.step(data)
                if terminated or truncated:
                    obs, _ = env.reset()
                remote.send((obs, reward, terminated, truncated, info))

            elif cmd == 'reset':
                obs, info = env.reset()
                remote.send((obs, info))

            elif cmd == 'close':
                env.close()
                remote.close()
                break

            elif cmd == 'get_valid_actions':
                if hasattr(env, 'get_valid_actions'):
                    valid = env.get_valid_actions()
                else:
                    valid = list(range(env.action_space.n))
                remote.send(valid)

        except KeyboardInterrupt:
            break

# wrapper para ejecutar multiples ambientes en paralelo
class ParallelEnv:
    def __init__(self, env_fn, num_envs=8):
        self.num_envs = num_envs
        self.waiting = False
        self.closed = False

        # crear pipes para comunicacion con workers
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(num_envs)])

        # iniciar procesos workers
        self.processes = []
        for work_remote, remote in zip(self.work_remotes, self.remotes):
            proc = Process(target=worker, args=(work_remote, remote, env_fn))
            proc.daemon = True
            proc.start()
            self.processes.append(proc)
            work_remote.close()

    # ejecuta acciones en todos los ambientes
    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))

        results = [remote.recv() for remote in self.remotes]
        obs, rewards, terminateds, truncateds, infos = zip(*results)

        return np.stack(obs), np.array(rewards), np.array(terminateds), \
               np.array(truncateds), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))

        results = [remote.recv() for remote in self.remotes]
        obs, infos = zip(*results)
        return np.stack(obs), infos

    def get_valid_actions(self):
        for remote in self.remotes:
            remote.send(('get_valid_actions', None))

        return [remote.recv() for remote in self.remotes]

    def close(self):
        if self.closed:
            return

        for remote in self.remotes:
            remote.send(('close', None))

        for proc in self.processes:
            proc.join()

        self.closed = True

    def __del__(self):
        if not self.closed:
            self.close()
