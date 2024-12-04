from django.apps import AppConfig
from .prediction import ResNetImagePredictor


class ImageClassificationConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'image_classification'
    predictor = ResNetImagePredictor()
