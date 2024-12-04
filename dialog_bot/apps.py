from django.apps import AppConfig
from .dialog_bot import DialogBotRuGPTSmall


class DialogBotConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'dialog_bot'
    dialog_bot = DialogBotRuGPTSmall()
