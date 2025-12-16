from flask import Flask, render_template, request
import cohere
import json
import pickle
#Leemos el.env
from dotenv import load_dotenv
load_dotenv() #se le pone la ruta al .env

#Leer la variable de entorno cohere_Api_key
import os
cohere_api_key = os.getenv("COHERE_API_KEY")


img_url = "https://w7.pngwing.com/pngs/935/387/png-transparent-man-portrait-line-art-male-human-people-person-vintage-retro-old.png"


co = cohere.ClientV2(api_key="xmCanDMQEBJuFOMdwyfFIcaLUsVKTc6Y3rYF56UH")


try:
    with open("modern.pkl", "rb") as f:
        read_rf = pickle.load(f)
except FileNotFoundError:
    read_rf = None

app = Flask(__name__)

def get_features_from_image(img_url):
    prompt = """
    Eres un modelo que recibe una imagen de entrada y tu ÚNICA tarea es devolver SIEMPRE un JSON con tres campos numéricos: "sex", "age" y "class".
    INSTRUCCIONES IMPORTANTES:
    1. FORMATO DE SALIDA
    - Debes responder SIEMPRE y SOLO con un objeto JSON válido, sin texto adicional antes ni después.
    - El formato exacto debe ser:
   {
        "sex": 0,
        "age": 35,
        "class": 2
    }
    - No incluyas comentarios, explicaciones, texto libre ni ningún otro campo.
    - Usa SIEMPRE comillas dobles en las claves ("sex", "age", "class").
    - Los tres valores deben ser números enteros (sin comillas).
    2. SIGNIFICADO DE LOS CAMPOS
    - "sex": entero que representa el sexo de la persona principal de la imagen.
        * 0 = mujer
        * 1 = hombre
    - "age": entero que representa una estimación de la edad (en años) de la persona principal.
    - "class": entero que representa una clase de camarote del Titanic:
        * 1 = primera
        * 2 = segunda
        * 3 = tercera
    3. CUÁNDO HAY UNA PERSONA CLARA EN LA IMAGEN
    - Si ves claramente una persona que parece ser la protagonista de la imagen:
        * Estima su sexo ("sex") según lo que más probable te parezca (0 o 1).
        * Estima su edad ("age") como un número entero razonable.
        * Estima su "class" (1, 2 o 3) usando tu mejor intuición a partir de su ropa, contexto, entorno, etc. No tiene que ser real, solo razonable.
    4. CUÁNDO HAY VARIAS PERSONAS, NINGUNA O ES MUY DIFÍCIL
    - Si hay varias personas:
        * Escoge una persona que parezca protagonista (por ejemplo, la más centrada o más cercana a la cámara) y decide los tres valores para esa persona.
    - Si no se ve claramente ninguna persona o la imagen es demasiado confusa:
        * INVÉNTATE valores razonables para "sex", "age" y "class".
    - Si tienes dudas sobre cualquiera de los campos:
        * AUN ASÍ debes elegir un valor y devolverlo. Nunca uses null, ni -1, ni dejes el campo vacío.
    5. REGLA MÁS IMPORTANTE
    - PASE LO QUE PASE debes devolver SIEMPRE un JSON con este esquema exacto:
    {
        "sex": <entero 0 o 1>,
        "age": <entero>,
        "class": <entero 1, 2 o 3>
    }
    - No añadas más campos.
    - No devuelvas texto adicional.
    - No expliques tus decisiones.
    - Aunque no estés seguro, elige los valores más razonables que puedas o invéntalos.
    """
    response = co.chat(
        model="command-a-vision-07-2025",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            # Can be either a base64 data URI or a web URL.
                            "url": img_url,
                            "detail": "auto"
                        }
                    }
                ]
            }
        ]
    )
    
    return json.loads(response.message.content[0].text)

@app.route("/", methods = ['GET']) #"/" --> endpoint  
def home():
    return render_template("frontend.html")

@app.route("/inicio", methods = ["POST", "GET"])
def inicio():
    prediction = None
    if request.method == "POST":
        img_url = request.form.get("img_url")
        features_dict = get_features_from_image(img_url)
        age = features_dict["age"] 
        sex = features_dict["sex"] 
        clase = features_dict["class"] 

        survived = "Sobrevivio" if read_rf.predict([[age,sex,clase]])[0] else ("murio")
        prediction = f"Para la persona con age {age}, sexo {sex} y clase {clase}, la predicción es de {survived}"

    return render_template("frontendresultado.html", nombre= "hola", poema = prediction, img_url = img_url)

if __name__ == "__main__":
    app.run(debug = True, host = "localhost", port  = 5000)