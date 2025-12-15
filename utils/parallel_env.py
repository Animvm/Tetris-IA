import multiprocessing as mp
from multiprocessing import Process, Pipe
import numpy as np

def worker(remote, parent_remote, env_fn):
    """
    Worker process para ejecutar un environment en paralelo.
    Se comunica con el proceso principal via pipes.

    Args:
        remote: pipe para comunicacion con proceso principal
        parent_remote: pipe del padre (se cierra en worker)
        env_fn: funcion que retorna una instancia del environment
    """
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

class ParallelEnv:
    """
    Wrapper para ejecutar multiples environments en paralelo.
    Utiliza multiprocessing para crear workers independientes.

    Beneficios:
    - Acelera coleccion de experiencia
    - Mejor uso de CPU multi-core
    - Permite batch inference en GPU
    """

    def __init__(self, env_fn, num_envs=8):
        """
        Args:
            env_fn: funcion que retorna environment (lambda: TetrisEnv(...))
            num_envs: numero de environments paralelos (ajustar segun CPU/GPU)
        """
        self.num_envs = num_envs
        self.waiting = False
        self.closed = False

        # Crear pipes para comunicacion bidireccional
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(num_envs)])

        # Iniciar worker processes
        self.processes = []
        for work_remote, remote in zip(self.work_remotes, self.remotes):
            proc = Process(target=worker, args=(work_remote, remote, env_fn))
            proc.daemon = True
            proc.start()
            self.processes.append(proc)
            work_remote.close()

    def step(self, actions):
        """
        Ejecuta step en todos los environments en paralelo.

        Args:
            actions: lista de acciones (length = num_envs)

        Returns:
            observations: array (num_envs, *obs_shape)
            rewards: array (num_envs,)
            terminateds: array (num_envs,)
            truncateds: array (num_envs,)
            infos: tuple de dicts
        """
        # Enviar comando a todos los workers
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))

        # Recibir resultados
        results = [remote.recv() for remote in self.remotes]
        obs, rewards, terminateds, truncateds, infos = zip(*results)

        return np.stack(obs), np.array(rewards), np.array(terminateds), \
               np.array(truncateds), infos

    def reset(self):
        """
        Reset todos los environments.

        Returns:
            observations: array (num_envs, *obs_shape)
            infos: tuple de dicts
        """
        for remote in self.remotes:
            remote.send(('reset', None))

        results = [remote.recv() for remote in self.remotes]
        obs, infos = zip(*results)
        return np.stack(obs), infos

    def get_valid_actions(self):
        """
        Obtiene acciones validas de todos los environments.

        Returns:
            lista de listas con acciones validas por environment
        """
        for remote in self.remotes:
            remote.send(('get_valid_actions', None))

        return [remote.recv() for remote in self.remotes]

    def close(self):
        """Cierra todos los workers y libera recursos."""
        if self.closed:
            return

        for remote in self.remotes:
            remote.send(('close', None))

        for proc in self.processes:
            proc.join()

        self.closed = True

    def __del__(self):
        """Asegurar que workers se cierren al destruir objeto."""
        if not self.closed:
            self.close()
