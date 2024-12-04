from django.db import models

class Task(models.Model):
    task_name = models.CharField(max_length=50)

    def __str__(self):
        return self.task_name

class InteractionML(models.Model):
    task = models.ForeignKey(Task, on_delete=models.CASCADE)
    input_data = models.CharField(max_length=200)
    output_data = models.CharField(max_length=200)

    def __str__(self):
        return (f"input:{self.input_data}, outuput:{self.output_data}")
