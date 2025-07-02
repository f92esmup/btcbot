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
            'text': message,
            'parse_mode': 'Markdown'
        }
        try:
            response = requests.post(self.api_url, data=payload, timeout=5)
            response.raise_for_status()  # Lanza una excepción para códigos de error HTTP
            print(f"Mensaje de Telegram enviado exitosamente.")
        except requests.exceptions.RequestException as e:
            print(f"Error al enviar mensaje de Telegram: {e}")
