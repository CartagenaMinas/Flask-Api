from flask import  Flask, request #importamos
from flask_cors import CORS
import torch
import numpy as np
from PIL import Image


app=Flask(__name__) #Instanciamos nuestra aplicacion
CORS(app)

model=torch.jot.load("model.zip")#Cuando carge el api cargara el modelo


@app.route('/')
def hello_world():
    return "Hello, World"


@app.route('/predict', methods=['POST']) #El http para cambiar de get a post enviar y recibir 
def predict():
    #load image
    img=Image.open(request.files["file"].stream).convert( #le enviamos la foto con el request y la transformamos
        "RGB").remap_palette((224,224)) #Le rescalo a la medida del entrenamiento
    img=np.array(img)#lo paso a array de numpy
    img=torch.FloatTensor(img.transpose((2,0,1))/255) #le pongo sus dimensiones y normalizo

    #get predictions
    preds=model(img.unsqueeze(0)).squeeze()#Le aumentamos una dimension para un bach de 1,despues le quitamos y nos quedamos con la prediccion
    probas=torch.softmax(preds,axis=0)#Lo convertimos en probabilidad
    ix=torch.argmax(probas,axis=0)#El mas grande es su clase

    return{
        "label": model.labels[ix],
        "score": probas[ix].item()
    }






if __name__ == "__main__":
    app.run()