from django.db import models
from bert_classifier.models import Task


class Image(models.Model):
    task = models.ForeignKey(Task, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='images')
    image_class = models.CharField(max_length=100)
