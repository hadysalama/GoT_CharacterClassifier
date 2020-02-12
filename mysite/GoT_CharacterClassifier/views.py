'''
@author Hady Salama
Created on: 7/7/2019
Using Python 3.72, Django 2.2.3, fastai 1.0.54 on Pytorch 1.1

'''
from django.shortcuts import render
import fastai.vision as Vision
import os

#Loads Model on start.
module_dir = os.path.dirname(__file__)  # get current directory
file_path = os.path.join(module_dir)
learn = Vision.load_learner(file_path,'export.pkl')

def index(request):
    #Renders main page.

    return render(request, 'GoT_CharacterClassifier/index.html')

def result(request):
    #Takes the image and runs it through the exported model via Pytorch CPU processing.
    
    print("The values are " + str(request.POST.keys()))

    if request.POST.get('file') != "":
        img = Vision.open_image(request.FILES.get('file'))
        pred_class, pred_idx, outputs = learn.predict(img)

        print()
        print("Predicted class is: " + str(pred_class))
        print("Predicted index is: " + str(pred_idx))
        print("Predicted outputs are" + str(outputs))
        print()

        if(str(pred_class) == "daenerys"):
            char = "Your character is Danaerys Targaryen."
        elif(str(pred_class) == "sansa"):
            char = "Your character is Sansa Stark."
        elif(str(pred_class) == "tyrion"):
            char = "Your character is Tyrion Lannister."
        else:
            char = "Your character is not recognized by the current model."
        
        context = {"character": char}
    else:
        context = {"character": "Please submit an image."}

    return render(request, 'GoT_CharacterClassifier/index.html', context)

