from flask import  Flask #importamos
from flask_cors import CORS


app=Flask(__name__) #Instanciamos nuestra aplicacion
CORS(app)

@app.route('/')
def hello_world():
    return "Hello, World"


@app.route('/predict')
def predict():
    return "predicciones"

if __name__ == "__main__":
    app.run()