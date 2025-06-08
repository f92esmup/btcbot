"""
Funciones de utilidad para validación de datos.
"""

from datetime import datetime


def validate_date_format(date_string: str) -> bool:
    """
    Valida que la fecha tenga el formato correcto.
    
    Args:
        date_string (str): Fecha en formato YYYY-MM-DD
        
    Returns:
        bool: True si es válida, False en caso contrario
    """
    try:
        datetime.strptime(date_string, '%Y-%m-%d')
        return True
    except ValueError:
        return False
