import requests


class TelegramNotifier:
    """
    Envía notificaciones a un chat de Telegram a través de un bot.
    """
    def __init__(self, bot_token: str, chat_id: str):
        """
        Inicializa el notificador de Telegram.
        
        Args:
            bot_token (str): El token del bot de Telegram.
            chat_id (str): El ID del chat al que se enviarán los mensajes.
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        print(f"TelegramNotifier inicializado para el chat_id: {self.chat_id}")
    
    def send_message(self, message: str):
        """
        Envía un mensaje de texto al chat de Telegram configurado.
        
        Args:
            message (str): El texto del mensaje a enviar.
        """
        payload = {
            'chat_id': self.chat_id,
            'text': message
        }
        try:
            response = requests.post(self.api_url, data=payload, timeout=5)
            response.raise_for_status()  # Lanza una excepción para códigos de error HTTP
            print(f"Mensaje de Telegram enviado exitosamente.")
        except requests.exceptions.RequestException as e:
            print(f"Error al enviar mensaje de Telegram: {e}")

    def notify_stop_loss_execution(self, symbol: str, position_type: str, entry_price: float, 
                                   exit_price: float, pnl: float, roe: float):
        """
        Envía una notificación detallada cuando se ejecuta un stop-loss.
        
        Args:
            symbol (str): El símbolo del trading (ej. BTCUSDT)
            position_type (str): Tipo de posición (LARGO/CORTO)
            entry_price (float): Precio de entrada
            exit_price (float): Precio de salida (stop-loss)
            pnl (float): Profit/Loss absoluto
            roe (float): Return on Equity en porcentaje
        """
        emoji = "🔴" if pnl < 0 else "🟢"
        message = (
            f"⚠️ STOP-LOSS EJECUTADO {emoji}\n\n"
            f"Símbolo: {symbol}\n"
            f"Posición: {position_type}\n"
            f"Entrada: ${entry_price:,.2f}\n"
            f"Salida: ${exit_price:,.2f}\n"
            f"PnL: ${pnl:,.2f}\n"
            f"ROE: {roe*100:.2f}%"
        )
        self.send_message(message)

    def notify_emergency_closure(self, symbol: str, position_type: str, error_details: str):
        """
        Envía una alerta crítica cuando se produce un cierre de emergencia.
        
        Args:
            symbol (str): El símbolo del trading
            position_type (str): Tipo de posición (LARGO/CORTO)
            error_details (str): Detalles del error que causó el cierre de emergencia
        """
        message = (
            f"🚨 ¡ALERTA CRÍTICA! 🚨\n\n"
            f"Cierre de Emergencia en {symbol}\n"
            f"Posición: {position_type}\n"
            f"Causa: Fallo al colocar Stop-Loss\n"
            f"Error: {error_details}\n\n"
            f"⚠️ Se requiere intervención manual para verificar el estado de la cuenta."
        )
        self.send_message(message)
