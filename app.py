from flask import Flask, render_template, jsonify
import pickle
from model import transform_input, getFoodName, initialize
import numpy as np

app = Flask(__name__)
ml_model = pickle.load(open('trained-model.pkl','rb'))
ingredients, courses, X_dict, X, y = initialize()

@app.get("/")
def index():
    return render_template("homepage.html")

@app.get('/api')
def predict():
    my_ingredients = ["biryani masala", "yoghurt", "sugar", "flour", "green chillies", "rice", "chicken", "milk"]
    # my_ingredients = ["yoghurt", "sugar", "flour", "rice", "milk", "almonds", "sugar syrup", "dry fruits"]
    course = ["starter"]
    encoded_input = transform_input(my_ingredients, course, courses, ingredients)
    probabilities = ml_model.predict_proba(encoded_input)
    predictions = np.argsort(-probabilities).flatten()
    top_5_foods = getFoodName(predictions[0:5], X_dict)
    return render_template("homepage.html", foods=top_5_foods)


@app.get("/ingredients")
def get_ingredients():
    response = { 
        "ingredients": ingredients
    }
    return jsonify(response)
if __name__ == "__main__":
    app.run(debug=True)
