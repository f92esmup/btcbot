"""
Tests para el módulo de normalización.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# Añadir el directorio raíz al path para importaciones
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.normalization import Normalization


class TestNormalization:
    """Clase de tests para la normalización de datos."""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Crea un DataFrame de muestra para testing."""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='1h')
        
        data = {
            'Open': np.random.uniform(50000, 60000, 100),
            'High': np.random.uniform(50500, 60500, 100),
            'Low': np.random.uniform(49500, 59500, 100),
            'Close': np.random.uniform(50000, 60000, 100),
            'Volume': np.random.uniform(1000000, 5000000, 100),
            'EMA_20': np.random.uniform(50000, 60000, 100),
            'RSI_14': np.random.uniform(20, 80, 100),
            'ATR_14': np.random.uniform(100, 1000, 100)
        }
        
        df = pd.DataFrame(data, index=dates)
        return df.astype('float32')
    
    @pytest.fixture
    def temp_scaler_path(self):
        """Crea un path temporal para el scaler."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield os.path.join(temp_dir, "test_scaler.pkl")
    
    def test_initialization(self, sample_dataframe):
        """Test de inicialización de la clase."""
        norm = Normalization(sample_dataframe)
        
        assert norm.dataframe is not None
        assert len(norm.dataframe) == len(sample_dataframe)
        assert norm.scaler is None
        assert norm.feature_columns is None
        assert norm.scaler_type == "MinMaxScaler"
        assert norm.feature_range == (0, 1)
    
    def test_initialization_empty_dataframe(self):
        """Test de inicialización con DataFrame vacío."""
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="El DataFrame proporcionado está vacío"):
            Normalization(empty_df)
    
    def test_prepare_features(self, sample_dataframe):
        """Test de preparación de características."""
        norm = Normalization(sample_dataframe)
        norm._prepare_features()
        
        # Verificar que se identificaron las columnas numéricas
        assert norm.feature_columns is not None
        assert len(norm.feature_columns) == len(sample_dataframe.columns)
        assert all(col in sample_dataframe.columns for col in norm.feature_columns)
    
    def test_prepare_features_with_nans(self, sample_dataframe):
        """Test de preparación con valores NaN."""
        # Añadir algunos NaN
        sample_dataframe.iloc[0:5, 0] = np.nan
        
        norm = Normalization(sample_dataframe)
        initial_length = len(norm.dataframe)
        norm._prepare_features()
        
        # Verificar que se eliminaron las filas con NaN
        assert len(norm.dataframe) < initial_length
        assert not norm.dataframe.isna().any().any()
    
    def test_fit_scaler(self, sample_dataframe):
        """Test de ajuste del scaler."""
        norm = Normalization(sample_dataframe)
        norm._prepare_features()
        norm._fit_scaler()
        
        # Verificar que el scaler fue creado y ajustado
        assert norm.scaler is not None
        assert isinstance(norm.scaler, MinMaxScaler)
        assert hasattr(norm.scaler, 'data_min_')
        assert hasattr(norm.scaler, 'data_max_')
        assert len(norm.scaler.data_min_) == len(norm.feature_columns)
    
    def test_transform_datasets(self, sample_dataframe):
        """Test de transformación de datos."""
        norm = Normalization(sample_dataframe)
        norm._prepare_features()
        norm._fit_scaler()
        
        transformed_df = norm._transform_datasets()
        
        # Verificar forma y estructura
        assert transformed_df.shape == norm.dataframe[norm.feature_columns].shape
        assert list(transformed_df.columns) == norm.feature_columns
        assert transformed_df.index.equals(norm.dataframe.index)
        
        # Verificar que los valores están en el rango [0, 1]
        assert (transformed_df >= 0).all().all()
        assert (transformed_df <= 1).all().all()
        
        # Verificar que al menos algunos valores están cerca de los extremos
        assert transformed_df.min().min() < 0.1  # Algunos valores cerca de 0
        assert transformed_df.max().max() > 0.9  # Algunos valores cerca de 1
    
    def test_main_method(self, sample_dataframe, temp_scaler_path):
        """Test del método principal."""
        # Configurar path temporal para el scaler
        norm = Normalization(sample_dataframe)
        norm.scaler_path = temp_scaler_path
        
        # Ejecutar proceso completo
        normalized_df, scaler = norm.main()
        
        # Verificar resultados
        assert normalized_df is not None
        assert scaler is not None
        assert isinstance(scaler, MinMaxScaler)
        
        # Verificar que el scaler se guardó
        assert os.path.exists(temp_scaler_path)
        
        # Verificar normalización
        assert (normalized_df >= 0).all().all()
        assert (normalized_df <= 1).all().all()
    
    def test_load_scaler(self, sample_dataframe, temp_scaler_path):
        """Test de carga de scaler guardado."""
        # Crear y guardar un scaler
        norm = Normalization(sample_dataframe)
        norm.scaler_path = temp_scaler_path
        norm._prepare_features()
        norm._fit_scaler()
        norm._save_scaler()
        
        # Cargar el scaler
        loaded_scaler = Normalization.load_scaler(temp_scaler_path)
        
        # Verificar que se cargó correctamente
        assert loaded_scaler is not None
        assert isinstance(loaded_scaler, MinMaxScaler)
        assert np.array_equal(loaded_scaler.data_min_, norm.scaler.data_min_)
        assert np.array_equal(loaded_scaler.data_max_, norm.scaler.data_max_)
    
    def test_load_scaler_nonexistent_file(self):
        """Test de carga de scaler con archivo inexistente."""
        with pytest.raises(FileNotFoundError):
            Normalization.load_scaler("/path/to/nonexistent/scaler.pkl")
    
    def test_get_feature_info(self, sample_dataframe):
        """Test de obtención de información de características."""
        norm = Normalization(sample_dataframe)
        norm._prepare_features()
        norm._fit_scaler()
        
        info = norm.get_feature_info()
        
        # Verificar información básica
        assert 'num_features' in info
        assert 'feature_columns' in info
        assert 'scaler_type' in info
        assert 'feature_range' in info
        assert 'scaler_fitted' in info
        
        assert info['num_features'] == len(norm.feature_columns)
        assert info['feature_columns'] == norm.feature_columns
        assert info['scaler_type'] == 'MinMaxScaler'
        assert info['scaler_fitted'] is True
    
    def test_validate_normalization(self, sample_dataframe):
        """Test de validación de normalización."""
        norm = Normalization(sample_dataframe)
        norm._prepare_features()
        norm._fit_scaler()
        
        # Crear datos normalizados válidos
        normalized_data = np.random.uniform(0, 1, (100, len(norm.feature_columns)))
        normalized_df = pd.DataFrame(
            normalized_data,
            columns=norm.feature_columns,
            index=sample_dataframe.index
        )
        
        # Esto no debería lanzar errores
        norm._validate_normalization(normalized_df)
    
    def test_consistency_after_load(self, sample_dataframe, temp_scaler_path):
        """Test de consistencia después de cargar scaler."""
        # Crear scaler original
        norm1 = Normalization(sample_dataframe)
        norm1.scaler_path = temp_scaler_path
        normalized_df1, scaler1 = norm1.main()
        
        # Cargar scaler y aplicar a nuevos datos
        loaded_scaler = Normalization.load_scaler(temp_scaler_path)
        
        # Usar el scaler cargado para transformar los mismos datos
        feature_data = sample_dataframe[norm1.feature_columns].values
        normalized_data2 = loaded_scaler.transform(feature_data)
        
        # Verificar que los resultados son idénticos
        np.testing.assert_array_almost_equal(
            normalized_df1.values,
            normalized_data2,
            decimal=6
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
