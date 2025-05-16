    def _get_atr_value_optimized(self, current_market_data, close_price):
        """
        Método auxiliar para obtener el valor ATR con caché de índices.
        
        Args:
            current_market_data: Datos de mercado actuales
            close_price: Precio de cierre actual
            
        Returns:
            Valor ATR (desnormalizado)
        """
        # Cachear el índice ATR para evitar búsquedas repetidas
        if not hasattr(self, '_atr_idx_cache'):
            self._atr_idx_cache = self.feature_names.index('ATR_norm') if 'ATR_norm' in self.feature_names else -1
            
        atr_idx = self._atr_idx_cache
        
        # Si tenemos el índice, extraer el valor
        if atr_idx >= 0:
            # Des-normalizar si ATR_norm = ATR / Close
            atr_value = current_market_data[-1, atr_idx] * close_price
            return atr_value
        else:
            # Valor por defecto
            return close_price * 0.01  # 1% del precio como aproximación
