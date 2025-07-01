import torch
import numpy as np
from src.agente.agent import TransformerSACAgent
from src.training.run_manager import RunManager


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
        env_config = main_config.get('environment', {})
        agent_config = main_config.get('agent', {})

        # 2. Cargar scaler del entrenamiento usando el run_manager inyectado
        self.scaler = self.run_manager.load_scaler()

        # 3. Inferir parámetros del scaler y run_config
        # Asegurarse de que el scaler esté ajustado
        if not hasattr(self.scaler, 'n_features_in_'):
             raise ValueError("El scaler cargado no parece estar ajustado (no tiene 'n_features_in_').")
        
        num_total_features = self.scaler.n_features_in_
        sequence_length = env_config['ventana_observacion_size']
        # El número de características del portfolio es fijo y conocido
        self.portfolio_features = 4
        # Las características de mercado son el total menos las del portfolio
        self.market_features = num_total_features - self.portfolio_features
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
        model_prefix = f"{self.run_id}/best_model"
        self.run_manager.load_agent_from_checkpoint(
            agent=self.agent,
            checkpoint_prefix=model_prefix,
            reset_optimizers=True
        )
        
        # 6. Finalizar
        self.agent.eval_mode()
        
        print(f"✅ Agente cargado exitosamente desde run_id: {self.run_id}")
        print(f"   - Arquitectura inferida del scaler: {num_total_features} features total.")
    
    def _parse_observation(self, observation: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parsea la observación numpy en tensores market_data y portfolio_data.
        
        Args:
            observation: Array numpy con la observación completa
            
        Returns:
            Tupla con (market_data_tensor, portfolio_data_tensor)
        """
        # La observación tiene la estructura: [ventana_flat, portfolio_features]
        # ventana_flat tiene shape: (sequence_length * market_features,)
        # portfolio_features tiene shape: (portfolio_features,)
        
        market_data_flat_size = self.sequence_length * self.market_features
        
        # Separar market data y portfolio data
        market_data_flat = observation[:market_data_flat_size]
        portfolio_data = observation[market_data_flat_size:]
        
        # Reshape market data de flat a (sequence_length, market_features)
        market_data = market_data_flat.reshape(self.sequence_length, self.market_features)
        
        # Convertir a tensores de PyTorch y agregar dimensión de batch
        market_data_tensor = torch.from_numpy(market_data).float().unsqueeze(0).to(self.device)
        portfolio_data_tensor = torch.from_numpy(portfolio_data).float().unsqueeze(0).to(self.device)
        
        return market_data_tensor, portfolio_data_tensor
    
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
        market_data, portfolio_data = self._parse_observation(observation)
        
        # Obtener acción determinista del agente
        action = self.agent.select_action(
            market_data=market_data,
            portfolio_data=portfolio_data,
            deterministic=True
        )
        
        # Devolver la acción como float (select_action devuelve numpy array)
        return float(action[0])
