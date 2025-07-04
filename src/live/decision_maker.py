import torch
import numpy as np
from src.agente.agent import TransformerSACAgent
from src.agente.networks import ActorNetwork, CriticNetwork
from src.training.checkpoint_manager import CheckpointManager
from src.utils.observation_parser import parse_observation
from src.configuration import AgentConfig


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

        if self.scaler is None:
            raise ValueError("El scaler inyectado no puede ser None")

        main_config = self.run_config.get('config', {})
        self.env_config = main_config.get('environment', {})
        agent_config_dict = main_config.get('agent', {})
        agent_config = AgentConfig(**agent_config_dict)

        if not hasattr(self.scaler, 'n_features_in_'):
             raise ValueError("El scaler inyectado no parece estar ajustado (no tiene 'n_features_in_').")

        market_features = self.scaler.n_features_in_
        sequence_length = self.env_config['ventana_observacion_size']
        portfolio_features = agent_config.architecture.portfolio_features
        action_dim = 1

        transformer_config = agent_config.transformer
        mlp_hidden_dims = agent_config.mlp_heads.hidden_dims

        actor = ActorNetwork(
            market_features=market_features,
            portfolio_features=portfolio_features,
            transformer_config=transformer_config,
            mlp_hidden_dims=mlp_hidden_dims,
            action_dim=action_dim,
            agent_config=agent_config
        )

        critic_1 = CriticNetwork(
            market_features=market_features,
            portfolio_features=portfolio_features,
            action_dim=action_dim,
            transformer_config=transformer_config,
            mlp_hidden_dims=mlp_hidden_dims,
            agent_config=agent_config
        )

        critic_2 = CriticNetwork(
            market_features=market_features,
            portfolio_features=portfolio_features,
            action_dim=action_dim,
            transformer_config=transformer_config,
            mlp_hidden_dims=mlp_hidden_dims,
            agent_config=agent_config
        )

        observation_space_shape = (sequence_length * market_features + portfolio_features,)
        action_space_shape = (1,)
        
        self.agent = TransformerSACAgent(
            actor=actor,
            critic_1=critic_1,
            critic_2=critic_2,
            observation_space_shape=observation_space_shape,
            action_space_shape=action_space_shape,
            config_override=agent_config,
            device=self.device,
            is_distributed=False
        )
        
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
        
        market_data, portfolio_data = parse_observation(
            observation, 
            self.env_config, 
            self.agent.portfolio_features, 
            self.device
        )
        
        action = self.agent.select_action(
            market_data=market_data,
            portfolio_data=portfolio_data,
            deterministic=True
        )
        
        return float(action[0])
