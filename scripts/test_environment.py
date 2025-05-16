import os
import sys
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import logging
import traceback

# Asegura que se puedan importar los módulos desde src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"Directorio raíz añadido: {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}")

# Importa el entorno
from src.environments import TradingEnvironment

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('TestEnv')

def random_agent_test(env, episodes=5, render=False):
    """
    Prueba el entorno con un agente aleatorio.
    
    Args:
        env: El entorno Gymnasium
        episodes: Número de episodios a ejecutar
        render: Si True, renderiza cada paso
    """
    # Estadísticas por episodio
    episode_returns = []
    episode_lengths = []
    episode_equity_curves = []
    
    for episode in range(episodes):
        logger.info(f"\n--- Episodio {episode + 1}/{episodes} ---")
        
        # Reinicia el entorno
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
            
            # Ejecuta un paso
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
    
    return episode_returns, episode_lengths, episode_equity_curves

def plot_results(returns, lengths, equity_curves):
    """
    Visualiza los resultados de la prueba.
    
    Args:
        returns: Lista de retornos por episodio
        lengths: Lista de longitudes de episodio
        equity_curves: Lista de curvas de equity por episodio
    """
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
    
    plt.tight_layout()
    plt.savefig('random_agent_results.png')
    plt.show()

if __name__ == "__main__":
    try:
        print("Iniciando test del entorno de trading...")
        
        # Verifica la estructura de directorios
        print(f"Directorio actual: {os.getcwd()}")
        print(f"Archivos en src/environments: {os.listdir('src/environments') if os.path.exists('src/environments') else 'Directorio no encontrado'}")
        
        # Crea el entorno directamente
        print("Creando entorno TradingEnvironment...")
        env = TradingEnvironment(render_mode='human')
        print(f"Entorno creado exitosamente: {env}")
        
        # Prueba con agente aleatorio
        print("Iniciando prueba con agente aleatorio...")
        random_agent_test(env, episodes=1, render=True)
        
        # Cierra el entorno
        print("Cerrando entorno...")
        env.close()
        print("Test completado exitosamente.")
        
    except Exception as e:
        print(f"Error durante la ejecución: {e}")
        print("Traza de error:")
        traceback.print_exc()
