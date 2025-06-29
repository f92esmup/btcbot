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
        self.agent = None
        
        self._load_agent()
    
    def _load_agent(self):
        """
        Método privado para cargar el agente desde el run_id especificado.
        """
        # 1. Cargar configuración y artefactos
        run_manager = RunManager()
        run_manager.set_run_context(self.run_id)
        run_config = run_manager.download_and_load_yaml_config(self.run_id)
        if run_config is None:
            raise ValueError(f"No se pudo cargar la configuración para el run_id: {self.run_id}")
        
        # El scaler se carga para obtener el número de características de la observación
        scaler = run_manager.load_scaler()
        
        # 2. Extraer parámetros de la arquitectura de forma robusta
        try:
            env_config = run_config['config_snapshot']['environment']
            agent_config_snapshot = run_config['config_snapshot']['agent']
            
            # El número total de características de la observación lo sabe el scaler
            num_total_features = scaler.n_features_in_
            sequence_length = env_config['ventana_observacion_size']
            
            # La forma de la observación es (L, num_features), donde L es la longitud de la secuencia
            observation_space_shape = (num_total_features,)
            
            # Las características del portfolio son fijas (4)
            portfolio_features = 4
            
            # Las características del mercado son el total menos las del portfolio
            market_features = num_total_features - portfolio_features
            
            # La forma del espacio de acción es fija (1)
            action_space_shape = (1,)
            
            # Usamos la configuración guardada en el snapshot para asegurar consistencia
            config_override = agent_config_snapshot
            
        except KeyError as e:
            raise ValueError(f"Parámetro de configuración faltante en el run_config.yaml: {e}")
        
        # 3. Instanciar el agente con los parámetros correctos
        self.agent = TransformerSACAgent(
            observation_space_shape=observation_space_shape,
            action_space_shape=action_space_shape,
            market_features=market_features,
            portfolio_features=portfolio_features,
            sequence_length=sequence_length,
            device=self.device,
            config_override=config_override,
            is_distributed=False
        )
        
        # 4. Cargar los pesos del modelo (usamos "best_model" como el objetivo estándar para live)
        model_prefix = f"{self.run_id}/best_model"
        run_manager.load_agent_from_checkpoint(
            agent=self.agent,
            checkpoint_prefix=model_prefix,
            reset_optimizers=True
        )
        
        # 5. Finalizar
        self.agent.eval_mode()
        
        # Guardar parámetros para parsear observaciones
        self.market_features = market_features
        self.portfolio_features = portfolio_features
        self.sequence_length = sequence_length
        
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
