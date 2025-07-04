import torch
import numpy as np
from src.agente.agent import TransformerSACAgent
from src.training.checkpoint_manager import CheckpointManager
from src.utils.observation_parser import parse_observation


class DecisionMaker:
    """
    Gestor para el agente de RL en modo live.
    Se encarga de cargar el modelo entrenado y proporcionar decisiones.
    """
    
    def __init__(self, scaler, checkpoint_manager: CheckpointManager, run_config: dict, device: torch.device, run_id: str):
        """
        Inicializa el DecisionMaker.

        Args:
            scaler: El scaler ya cargado para obtener información de características.
            checkpoint_manager (CheckpointManager): Instancia del gestor de checkpoints para cargar el modelo.
            run_config (dict): Configuración completa del run.
            device (torch.device): Dispositivo donde ejecutar el modelo (CPU/GPU).
            run_id (str): El ID del run de entrenamiento a utilizar.
        """
        self.scaler = scaler
        self.checkpoint_manager = checkpoint_manager
        self.run_config = run_config
        self.device = device
        self.run_id = run_id

        # Validar que el scaler inyectado no sea None
        if self.scaler is None:
            raise ValueError("El scaler inyectado no puede ser None")

        # 1. Extraer configuraciones del entorno y agente desde la clave 'config'
        main_config = self.run_config.get('config', {})
        self.env_config = main_config.get('environment', {})
        agent_config = main_config.get('agent', {})

        # 2. Asegurarse de que el scaler esté ajustado
        if not hasattr(self.scaler, 'n_features_in_'):
             raise ValueError("El scaler inyectado no parece estar ajustado (no tiene 'n_features_in_').")

        # 3. Determinar la arquitectura del agente desde la "Fuente de la Verdad"
        self.market_features = self.scaler.n_features_in_
        self.sequence_length = self.env_config['ventana_observacion_size']
        self.portfolio_features = agent_config.get('architecture', {}).get('portfolio_features', 4)

        # 4. Instanciar el agente con la configuración y arquitectura correctas
        observation_space_shape = (self.sequence_length * self.market_features + self.portfolio_features,)
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
        training_run_prefix = self.checkpoint_manager._get_training_run_prefix(self.run_id)
        model_prefix = f"{training_run_prefix}/{model_to_load}"
        print(f"Cargando modelo: {model_prefix}")

        self.checkpoint_manager.load_agent_from_checkpoint(
            agent=self.agent,
            checkpoint_prefix=model_prefix,
            reset_optimizers=False
        )
        
        # 6. Finalizar
        self.agent.eval_mode()
        
        print(f"✅ Agente cargado exitosamente desde run_id: {self.run_id}")
        print(f"   - Arquitectura del modelo cargada desde la configuración del run.")
    
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
            deterministic=True
        )
        
        # Devolver la acción como float (select_action devuelve numpy array)
        return float(action[0])
