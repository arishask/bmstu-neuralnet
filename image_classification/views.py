import os
from django.http import HttpResponseRedirect
from bert_classifier.models import Task
from django.shortcuts import get_object_or_404, render
from .apps import ImageClassificationConfig
from django.urls import reverse
from image_classification.forms import UserImage
from django.conf import settings


def index(request):
    tasks_list = Task.objects.all()
    context = {'tasks_list': tasks_list}
    return render(request, 'image_classification/index.html', context)

def detail(request, task_id):
    task = get_object_or_404(Task, pk=task_id)
    latest_predictions = task.image_set.order_by('-id')[:1][::-1]
    context = {
        'task': task,
        'latest_predictions': latest_predictions
    }

    if request.method == 'POST':
        form = UserImage(request.POST, request.FILES, initial={'task': task_id})
        if form.is_valid():
            form.save()
            context["image"] = form.instance
    else:
        form = UserImage(initial={'task': task_id})

    context["form"] = form

    return render(request, 'image_classification/detail.html', context)


def predict(request, task_id):
    task = get_object_or_404(Task, pk=task_id)

    if request.method == 'POST':
        form = UserImage(request.POST, request.FILES, initial={'task': task_id})
        if form.is_valid():
            form.instance.task = task
            form.save()
            image_path = os.path.join(settings.BASE_DIR,
                                      *[x for x in form.instance.image.url.split('/')])
            output = ImageClassificationConfig.predictor.predict(image_path)
            form.instance.image_class = output
            form.save()

    return HttpResponseRedirect(reverse('image_classification:detail', args=(task.id,)))