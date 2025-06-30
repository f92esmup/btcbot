import torch
import numpy as np
from src.agente.agent import TransformerSACAgent
from src.training.run_manager import RunManager


class DecisionMaker:
    """
    Gestor para el agente de RL en modo live.
    Se encarga de cargar el modelo entrenado y proporcionar decisiones.
    """
    
    def __init__(self, run_id: str, device: torch.device):
        """
        Inicializa el DecisionMaker con un run_id específico.
        
        Args:
            run_id: Identificador del run del cual cargar el modelo
            device: Dispositivo donde ejecutar el modelo (CPU/GPU)
        """
        self.run_id = run_id
        self.device = device
        
        # 1. Instanciar RunManager y establecer contexto
        run_manager = RunManager()
        run_manager.set_run_context(run_id)
        
        # 2. Cargar configuración del run
        self.run_config = run_manager.download_and_load_yaml_config(run_id)
        if self.run_config is None:
            raise ValueError(f"No se pudo cargar la configuración para el run_id: {run_id}")
        
        # 3. Cargar scaler del entrenamiento
        self.scaler = run_manager.load_scaler()
        
        # 4. Extraer configuraciones del entorno y agente
        env_config = self.run_config['config_snapshot']['environment']
        agent_config = self.run_config['config_snapshot']['agent']
        
        # 5. Inferir parámetros del scaler y run_config
        num_total_features = self.scaler.n_features_in_
        sequence_length = env_config['ventana_observacion_size']
        portfolio_features = 4  # Las características del portfolio son fijas
        market_features = num_total_features - portfolio_features
        
        # 6. Guardar parámetros inferidos como atributos de la clase
        self.market_features = market_features
        self.portfolio_features = portfolio_features
        self.sequence_length = sequence_length
        
        # 7. Instanciar el agente con config_override
        observation_space_shape = (num_total_features,)
        action_space_shape = (1,)
        
        self.agent = TransformerSACAgent(
            observation_space_shape=observation_space_shape,
            action_space_shape=action_space_shape,
            market_features=market_features,
            portfolio_features=portfolio_features,
            sequence_length=sequence_length,
            device=self.device,
            config_override=agent_config,
            is_distributed=False
        )
        
        # 8. Cargar checkpoint con model_prefix correcto
        model_prefix = f"{run_id}/best_model"
        run_manager.load_agent_from_checkpoint(
            agent=self.agent,
            checkpoint_prefix=model_prefix,
            reset_optimizers=True
        )
        
        # 9. Finalizar
        self.agent.eval_mode()
        
        print(f"✅ Agente cargado exitosamente desde run_id: {run_id}")
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
