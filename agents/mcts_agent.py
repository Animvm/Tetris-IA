import numpy as np
import random


class Nodo:
    def __init__(self, estado, accion=None, padre=None):
        self.estado = estado
        self.accion = accion
        self.padre = padre
        self.hijos = []
        self.visitas = 0
        self.valor = 0.0
        self.acciones = None
    
    def expandido(self):
        return self.acciones is not None and len(self.acciones) == 0
    
    def mejor_hijo(self, c=1.41):
        if not self.hijos:
            return None
        
        mejor = -float('inf')
        mejor_nodo = None
        
        for hijo in self.hijos:
            if hijo.visitas == 0:
                return hijo
            
            explotacion = hijo.valor / hijo.visitas
            exploracion = c * np.sqrt(np.log(self.visitas) / hijo.visitas)
            ucb = explotacion + exploracion
            
            if ucb > mejor:
                mejor = ucb
                mejor_nodo = hijo
        
        return mejor_nodo


def evaluar_tablero(tablero):
    altura = np.sum(tablero > 0, axis=0)
    altura_max = np.max(altura) if altura.size > 0 else 0
    huecos = 0
    for col in range(tablero.shape[1]):
        primera = -1
        for fila in range(tablero.shape[0]):
            if tablero[fila, col] > 0 and primera == -1:
                primera = fila
            if primera != -1 and tablero[fila, col] == 0:
                huecos += 1
    lineas = np.sum(np.all(tablero > 0, axis=1))
    return lineas * 100 - huecos * 20 - altura_max * 5


class MCTSAgent:
    def __init__(self, env, num_simulaciones=100, max_profundidad=10):
        self.env = env
        self.num_simulaciones = num_simulaciones
        self.max_profundidad = max_profundidad
    
    def obtener_acciones(self):
        """
        Obtener acciones validas usando el metodo del environment si esta disponible.
        Fallback al metodo original si no hay action masking.
        """
        if hasattr(self.env, 'get_valid_actions') and hasattr(self.env, 'use_action_masking') and self.env.use_action_masking:
            return self.env.get_valid_actions()
        else:
            # Metodo original: todas las combinaciones de columna y rotacion
            acciones = []
            for col in range(self.env.cols):
                for rot in range(4):
                    accion = col * 4 + rot
                    acciones.append(accion)
            return acciones
    
    def seleccionar_accion(self, obs):
        raiz = Nodo(estado=obs)
        raiz.acciones = self.obtener_acciones()
        
        for _ in range(self.num_simulaciones):
            nodo = raiz
            
            while nodo.expandido() and nodo.hijos:
                nodo = nodo.mejor_hijo()
            
            if not nodo.expandido() and nodo.acciones:
                nodo = self.expandir(nodo)
            
            recompensa = self.simular(nodo.estado)
            
            self.propagar(nodo, recompensa)
        
        if not raiz.hijos:
            return random.choice(self.obtener_acciones())
        
        mejor = max(raiz.hijos, key=lambda h: h.visitas)
        return mejor.accion
    
    def expandir(self, nodo):
        if not nodo.acciones:
            return nodo
        
        accion = random.choice(nodo.acciones)
        nodo.acciones.remove(accion)
        
        from envs.tetris_env import TetrisEnv
        env_temp = TetrisEnv(rows=self.env.rows, cols=self.env.cols)
        env_temp.board = nodo.estado.copy()
        env_temp.done = False
        nuevo_estado, _, terminado, _, _ = env_temp.step(accion)
        
        hijo = Nodo(estado=nuevo_estado, accion=accion, padre=nodo)
        if not terminado:
            hijo.acciones = self.obtener_acciones()
        else:
            hijo.acciones = []
        
        nodo.hijos.append(hijo)
        return hijo
    
    def simular(self, estado):
        from envs.tetris_env import TetrisEnv
        
        env_temp = TetrisEnv(rows=self.env.rows, cols=self.env.cols)
        env_temp.board = estado.copy()
        env_temp.done = False
        
        recompensa_total = 0
        profundidad = 0
        terminado = False
        acciones = self.obtener_acciones()
        
        while not terminado and profundidad < self.max_profundidad:
            mejor_accion = None
            mejor_eval = -float('inf')
            
            for _ in range(5):
                accion = random.choice(acciones)
                env_prueba = TetrisEnv(rows=self.env.rows, cols=self.env.cols)
                env_prueba.board = env_temp.board.copy()
                env_prueba.done = False
                nuevo_estado, rew, term, trunc, _ = env_prueba.step(accion)
                
                if not (term or trunc):
                    evaluacion = evaluar_tablero(nuevo_estado) + rew
                    if evaluacion > mejor_eval:
                        mejor_eval = evaluacion
                        mejor_accion = accion
            
            if mejor_accion is None:
                mejor_accion = random.choice(acciones)
            
            _, recompensa, term, trunc, _ = env_temp.step(mejor_accion)
            recompensa_total += recompensa
            terminado = term or trunc
            profundidad += 1
        
        return recompensa_total
    
    def propagar(self, nodo, recompensa):
        while nodo is not None:
            nodo.visitas += 1
            nodo.valor += recompensa
            nodo = nodo.padre