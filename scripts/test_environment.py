import os
import sys
import time
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import logging
import traceback
import psutil
import gc

# Asegura que se puedan importar los módulos desde src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"Directorio raíz añadido: {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}")

# Importa el entorno
from src.environments import TradingEnvironment
from src.utils.config import ConfigManager
from src.utils.logging_utils import setup_logger

# Configurar logging
logger = setup_logger('TestEnv')

class PerformanceMonitor:
    """
    Clase para monitorear el rendimiento de la ejecución.
    """
    def __init__(self):
        self.start_time = None
        self.start_memory = None
        self.step_times = []
        self.reset_times = []
    
    def start(self):
        """Inicia el monitoreo de rendimiento"""
        gc.collect()  # Forzar recolección de basura antes de medir
        self.start_time = time.time()
        self.start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        return self
    
    def log_step(self, start_step_time):
        """Registra el tiempo de un paso"""
        self.step_times.append(time.time() - start_step_time)
    
    def log_reset(self, start_reset_time):
        """Registra el tiempo de un reset"""
        self.reset_times.append(time.time() - start_reset_time)
    
    def end(self):
        """Finaliza el monitoreo y muestra las estadísticas"""
        end_time = time.time()
        end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        
        total_time = end_time - self.start_time
        memory_used = end_memory - self.start_memory
        
        logger.info("\n--- Estadísticas de Rendimiento ---")
        logger.info(f"Tiempo total de ejecución: {total_time:.2f} segundos")
        logger.info(f"Memoria usada: {memory_used:.2f} MB")
        
        if self.step_times:
            # Descartamos el primer 10% de los pasos para evitar medir caché inicial/warmup
            if len(self.step_times) > 10:
                warmup_cutoff = int(len(self.step_times) * 0.1)
                stabilized_steps = self.step_times[warmup_cutoff:]
            else:
                stabilized_steps = self.step_times
            
            avg_step_time = np.mean(stabilized_steps) * 1000
            steps_per_second = 1 / np.mean(stabilized_steps) if np.mean(stabilized_steps) > 0 else 0
            
            logger.info(f"Tiempo promedio por step(): {avg_step_time:.2f} ms")
            logger.info(f"Step más rápido: {np.min(self.step_times) * 1000:.2f} ms")
            logger.info(f"Step más lento: {np.max(self.step_times) * 1000:.2f} ms")
            logger.info(f"Pasos por segundo: {steps_per_second:.2f}")
            logger.info(f"Total de pasos medidos: {len(self.step_times)}")
        
        if self.reset_times:
            logger.info(f"Tiempo promedio por reset(): {np.mean(self.reset_times) * 1000:.2f} ms")
        
        return {
            "total_time": total_time,
            "memory_used_mb": memory_used,
            "avg_step_time_ms": np.mean(self.step_times) * 1000 if self.step_times else 0,
            "steps_per_second": 1 / np.mean(self.step_times) if self.step_times and np.mean(self.step_times) > 0 else 0,
            "step_times": self.step_times  # Para graficar
        }

def random_agent_test(env, episodes=5, render=False, benchmark=False):
    """
    Prueba el entorno con un agente aleatorio.
    
    Args:
        env: El entorno Gymnasium
        episodes: Número de episodios a ejecutar
        render: Si True, renderiza cada paso
        benchmark: Si True, mide el rendimiento
    """
    # Iniciar monitor de rendimiento si está activado benchmarking
    perf_monitor = PerformanceMonitor().start() if benchmark else None
    # Estadísticas por episodio
    episode_returns = []
    episode_lengths = []
    episode_equity_curves = []
    
    for episode in range(episodes):
        logger.info(f"\n--- Episodio {episode + 1}/{episodes} ---")
        
        # Reinicia el entorno (con medición si está en modo benchmark)
        if benchmark:
            reset_start_time = time.time()
            observation, info = env.reset()
            perf_monitor.log_reset(reset_start_time)
        else:
            observation, info = env.reset()
        
        # Variables para seguimiento
        terminated = False
        truncated = False
        total_reward = 0
        steps = 0
        equity_curve = [env.current_equity]  # Equity inicial
        
        # Ejecuta el episodio
        while not (terminated or truncated):
            # Acción aleatoria
            action = env.action_space.sample()
            
            # Ejecuta un paso (con medición si está en modo benchmark)
            if benchmark:
                step_start_time = time.time()
                observation, reward, terminated, truncated, info = env.step(action)
                perf_monitor.log_step(step_start_time)
            else:
                observation, reward, terminated, truncated, info = env.step(action)
            
            # Actualiza variables de seguimiento
            total_reward += reward
            steps += 1
            equity_curve.append(env.current_equity)
            
            # Renderiza si está configurado
            if render:
                env.render()
        
        # Guarda estadísticas del episodio
        episode_return = env.current_equity / env.initial_equity_episode - 1.0
        episode_returns.append(episode_return)
        episode_lengths.append(steps)
        episode_equity_curves.append(equity_curve)
        
        logger.info(f"Episodio {episode + 1} completado:")
        logger.info(f"Retorno: {episode_return * 100:.2f}%")
        logger.info(f"Pasos: {steps}")
        logger.info(f"Razón de terminación: {info.get('termination_reason', 'N/A')}")
        logger.info(f"Operaciones: {info['trades_count']}")
        logger.info(f"Operaciones ganadoras: {info['episode_stats']['profitable_trades']} "
                   f"({info['win_rate'] * 100:.1f}% de éxito)")
        logger.info("-------------------")
    
    # Imprime estadísticas generales
    logger.info("\n--- Estadísticas Generales ---")
    logger.info(f"Retorno promedio: {np.mean(episode_returns) * 100:.2f}%")
    logger.info(f"Retorno máximo: {np.max(episode_returns) * 100:.2f}%")
    logger.info(f"Retorno mínimo: {np.min(episode_returns) * 100:.2f}%")
    logger.info(f"Longitud promedio de episodio: {np.mean(episode_lengths):.1f} pasos")
    
    # Visualiza los resultados
    plot_results(episode_returns, episode_lengths, episode_equity_curves)
    
    # Finaliza monitoreo de rendimiento si está activado
    if benchmark:
        perf_stats = perf_monitor.end()
        return episode_returns, episode_lengths, episode_equity_curves, perf_stats
    else:
        return episode_returns, episode_lengths, episode_equity_curves

def plot_results(returns, lengths, equity_curves, perf_stats=None):
    """
    Visualiza los resultados de la prueba.
    
    Args:
        returns: Lista de retornos por episodio
        lengths: Lista de longitudes de episodio
        equity_curves: Lista de curvas de equity por episodio
        perf_stats: Estadísticas de rendimiento (opcional)
    """
    if perf_stats:
        plt.figure(figsize=(15, 15))  # Más grande para incluir gráfico de rendimiento
    else:
        plt.figure(figsize=(15, 10))
    
    # Gráfico 1: Retornos por episodio
    plt.subplot(2, 2, 1)
    plt.bar(range(1, len(returns) + 1), [r * 100 for r in returns])
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.xlabel('Episodio')
    plt.ylabel('Retorno (%)')
    plt.title('Retorno por Episodio')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Gráfico 2: Longitudes de episodio
    plt.subplot(2, 2, 2)
    plt.bar(range(1, len(lengths) + 1), lengths)
    plt.xlabel('Episodio')
    plt.ylabel('Pasos')
    plt.title('Longitud de Episodio')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Gráfico 3: Curvas de equity
    plt.subplot(2, 1, 2)
    for i, equity in enumerate(equity_curves):
        plt.plot(equity, label=f'Episodio {i+1}')
    plt.axhline(y=equity_curves[0][0], color='r', linestyle='--', alpha=0.3, label='Equity Inicial')
    plt.xlabel('Pasos')
    plt.ylabel('Equity ($)')
    plt.title('Curvas de Equity por Episodio')
    plt.legend()
    
    # Si tenemos estadísticas de rendimiento, añadimos un gráfico adicional
    if perf_stats:
        plt.subplot(2, 2, 3)
        step_times_ms = np.array(perf_stats.get('step_times', [])) * 1000  # convertir a ms
        if len(step_times_ms) > 0:  # Solo plotear si hay datos
            plt.plot(step_times_ms)
            plt.xlabel('Paso')
            plt.ylabel('Tiempo (ms)')
            plt.title('Tiempo de Ejecución por Paso')
            
            # Añadir texto con estadísticas de rendimiento
            stats_text = (
                f"Pasos por segundo: {perf_stats.get('steps_per_second', 0):.2f}\n"
                f"Tiempo promedio de step: {perf_stats.get('avg_step_time_ms', 0):.2f} ms\n"
                f"Memoria utilizada: {perf_stats.get('memory_used_mb', 0):.2f} MB\n"
                f"Tiempo total: {perf_stats.get('total_time', 0):.2f} s"
            )
            plt.figtext(0.5, 0.02, stats_text, ha='center', bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 5})
    
    plt.tight_layout()
    plt.savefig('random_agent_results.png')
    plt.show()

def run_benchmark(episodes=3, iterations=2):
    """
    Ejecuta un benchmark comparando diferentes configuraciones del entorno.
    
    Args:
        episodes: Número de episodios por prueba
        iterations: Número de iteraciones para promediar resultados
    """
    config_path = "src/config.yaml"
    logger.info("\n=== INICIANDO BENCHMARK DE RENDIMIENTO ===\n")
    
    # Resultados para almacenar estadísticas
    benchmark_results = {}
    
    # Crear entorno con la configuración centralizada
    logger.info(f"Creando entorno con configuración: {config_path}")
    env = TradingEnvironment(config_path=config_path)
    logger.info(f"Entorno creado: {env}")
    
    # Ejecutar benchmark de referencia
    logger.info("\n--- Ejecutando benchmark de RENDIMIENTO ---")
    _, _, _, perf_stats_ref = random_agent_test(env, episodes=episodes, render=False, benchmark=True)
    benchmark_results['actual'] = perf_stats_ref
    env.close()
    
    # Como las optimizaciones ya han sido aplicadas al código, 
    # podemos simplemente mostrar los resultados actuales
    logger.info("\nLas optimizaciones ya están aplicadas al código.")
    logger.info("Para comparar con la versión anterior, deberías haber ejecutado este")
    logger.info("test antes de aplicar las optimizaciones y guardar los resultados.")
    
    # Mostrar resultados actuales
    logger.info("\n=== RESULTADOS DE RENDIMIENTO ===")
    steps_per_sec = benchmark_results['actual'].get('steps_per_second', 0)
    avg_step_time = benchmark_results['actual'].get('avg_step_time_ms', 0)
    memory_used = benchmark_results['actual'].get('memory_used_mb', 0)
    
    logger.info(f"Pasos/segundo: {steps_per_sec:.2f}")
    logger.info(f"Tiempo promedio por paso: {avg_step_time:.2f} ms")
    logger.info(f"Memoria utilizada: {memory_used:.2f} MB")
    
    # Sugerencia para guardar resultados para comparación futura
    logger.info("\n=== NOTA PARA COMPARACIÓN ===")
    logger.info("Para comparar rendimiento, guarda estos valores:")
    logger.info(f"Referencia: {steps_per_sec:.2f} pasos/segundo, {avg_step_time:.2f} ms/paso, {memory_used:.2f} MB")
    
    return benchmark_results

if __name__ == "__main__":
    try:
        print("Iniciando test del entorno de trading...")
        
        # Verifica la estructura de directorios
        print(f"Directorio actual: {os.getcwd()}")
        print(f"Archivos en src/environments: {os.listdir('src/environments') if os.path.exists('src/environments') else 'Directorio no encontrado'}")
        
        # Parsear argumentos de línea de comando
        import argparse
        parser = argparse.ArgumentParser(description='Test del entorno de trading')
        parser.add_argument('--benchmark', action='store_true', help='Ejecutar benchmark de rendimiento')
        parser.add_argument('--episodes', type=int, default=1, help='Número de episodios para el test')
        parser.add_argument('--render', action='store_true', help='Renderizar el entorno')
        args = parser.parse_args()
        
        if args.benchmark:
            # Ejecutar benchmark comparativo
            run_benchmark(episodes=args.episodes)
        else:
            # Crea el entorno para test estándar con la configuración centralizada
            config_path = "src/config.yaml"
            print("Creando entorno TradingEnvironment con la configuración centralizada...")
            env = TradingEnvironment(config_path=config_path, render_mode='human' if args.render else None)
            print(f"Entorno creado exitosamente: {env}")
            
            # Prueba con agente aleatorio
            print("Iniciando prueba con agente aleatorio...")
            random_agent_test(env, episodes=args.episodes, render=args.render, benchmark=True)
            
            # Cierra el entorno
            print("Cerrando entorno...")
            env.close()
            print("Test completado exitosamente.")
        
    except Exception as e:
        print(f"Error durante la ejecución: {e}")
        print("Traza de error:")
        traceback.print_exc()
