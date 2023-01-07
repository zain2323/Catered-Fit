from flask import Flask, render_template, jsonify, request
import pickle
from model import transform_input, getFoodName, initialize, getFoodDetails
import numpy as np

app = Flask(__name__)
ml_model = pickle.load(open('trained-model.pkl','rb'))
ingredients, courses, X_dict, X, y = initialize()

@app.get("/")
def index():
    return render_template("homepage.html")

@app.post('/predict')
def predict():
    data = request.get_json()
    course = data["course"]
    my_ingredients = data["ingredients"]
    encoded_input = transform_input(my_ingredients, course, courses, ingredients)
    probabilities = ml_model.predict_proba(encoded_input)
    predictions = np.argsort(-probabilities).flatten()
    top_5_foods = getFoodName(predictions[0:6], X_dict)
    response = { "foods": top_5_foods }
    return jsonify(response)


@app.get("/ingredients")
def get_ingredients():
    response = { 
        "ingredients": ingredients
    }
    return jsonify(response)


@app.get("/food/<string:name>")
def get_food(name):
    food = getFoodDetails(name)
    name = (food.name.values[0]).capitalize()
    ingredients = (food.ingredients.values[0]).split(",")
    ingredients = list(map(str.strip, ingredients))
    ingredients = list(map(str.title, ingredients))
    return render_template("food.html", name=name, ingredients=ingredients)


if __name__ == "__main__":
    app.run(debug=True)
