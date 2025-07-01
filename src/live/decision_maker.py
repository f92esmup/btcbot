import torch
import numpy as np
from src.agente.agent import TransformerSACAgent
from src.training.run_manager import RunManager
from src.utils.observation_parser import parse_observation


class DecisionMaker:
    """
    Gestor para el agente de RL en modo live.
    Se encarga de cargar el modelo entrenado y proporcionar decisiones.
    """
    
    def __init__(self, run_manager: RunManager, run_config: dict, device: torch.device):
        """
        Inicializa el DecisionMaker.

        Args:
            run_manager (RunManager): Instancia del gestor del run para cargar artefactos.
            run_config (dict): Configuración completa del run.
            device (torch.device): Dispositivo donde ejecutar el modelo (CPU/GPU).
        """
        self.run_manager = run_manager
        self.run_config = run_config
        self.device = device
        self.run_id = run_manager.run_id

        # 1. Extraer configuraciones del entorno y agente desde la clave 'config'
        main_config = self.run_config.get('config', {})
        self.env_config = main_config.get('environment', {})
        agent_config = main_config.get('agent', {})

        # 2. Cargar scaler del entrenamiento usando el run_manager inyectado
        self.scaler = self.run_manager.load_scaler()

        # 3. Inferir parámetros del scaler y run_config
        # Asegurarse de que el scaler esté ajustado
        if not hasattr(self.scaler, 'n_features_in_'):
             raise ValueError("El scaler cargado no parece estar ajustado (no tiene 'n_features_in_').")
        
        num_total_features = self.scaler.n_features_in_
        self.market_features = num_total_features
        sequence_length = self.env_config['ventana_observacion_size']
        # Leer el número de características del portfolio desde la configuración del agente
        self.portfolio_features = agent_config.get('architecture', {}).get('portfolio_features', 4)
        # Las características de mercado son el total menos las del portfolio
        # self.market_features = (num_total_features - self.portfolio_features) // sequence_length
        self.sequence_length = sequence_length

        # 4. Instanciar el agente con config_override
        observation_space_shape = (sequence_length * self.market_features + self.portfolio_features,)
        action_space_shape = (1,)
        
        self.agent = TransformerSACAgent(
            observation_space_shape=observation_space_shape,
            action_space_shape=action_space_shape,
            market_features=self.market_features,
            portfolio_features=self.portfolio_features,
            sequence_length=self.sequence_length,
            config_override=agent_config,
            device=self.device,
            is_distributed=False
        )
        
        # 5. Cargar checkpoint con model_prefix correcto
        live_config = main_config.get('live_trading', {})
        model_to_load = live_config.get('default_model_to_load', 'best_model')
        model_prefix = f"{self.run_id}/{model_to_load}"
        print(f"Cargando modelo: {model_prefix}")

        self.run_manager.load_agent_from_checkpoint(
            agent=self.agent,
            checkpoint_prefix=model_prefix,
            reset_optimizers=True
        )
        
        # 6. Finalizar
        self.agent.eval_mode()
        
        print(f"✅ Agente cargado exitosamente desde run_id: {self.run_id}")
        print(f"   - Arquitectura inferida del scaler: {num_total_features} features total.")
    
    def get_action(self, observation: np.ndarray) -> float:
        """
        Obtiene una acción del agente basada en la observación proporcionada.
        
        Args:
            observation: Observación del entorno
            
        Returns:
            Acción a tomar
        """
        if self.agent is None:
            raise ValueError("El agente no ha sido cargado correctamente")
        
        # Parsear la observación en market_data y portfolio_data
        market_data, portfolio_data = parse_observation(observation, self.env_config, self.device)
        
        # Obtener acción determinista del agente
        action = self.agent.select_action(
            market_data=market_data,
            portfolio_data=portfolio_data,
            deterministic=False
        )
        
        # Devolver la acción como float (select_action devuelve numpy array)
        return float(action[0])
