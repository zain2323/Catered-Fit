from flask import Flask, render_template, jsonify, request
import pickle
from model import transform_input, get_images_from_id, getFoodName, initialize, getFoodDetails, get_similar_foods, get_food_idx_list
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
    ocassion = data["ocassion"]
    my_ingredients = data["ingredients"]
    encoded_input = transform_input(my_ingredients, course, ocassion, courses, ingredients)
    probabilities = ml_model.predict_proba(encoded_input)
    predictions = np.argsort(-probabilities).flatten()
    top_5_foods = getFoodName(predictions[0:6], X_dict)
    top_5_foods_ids = get_food_idx_list(top_5_foods)
    images = get_images_from_id(top_5_foods_ids)
    response = { "foods": top_5_foods, "images": images }
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
    image = food.images.values[0]
    ingredients = (food.ingredients.values[0]).split(",")
    ingredients = list(map(str.strip, ingredients))
    ingredients = list(map(str.title, ingredients))
    prep_time = int(food.prep_time.values[0])
    cook_time = int(food.cook_time.values[0])
    # Recommended foods based on the current viewing dish
    recommended_food, images = get_similar_foods(food.name.values[0])
    return render_template("food.html", name=name, image=image, ingredients=ingredients, prep_time=prep_time, cook_time=cook_time, recommended_food=recommended_food, images=images, len=len)


if __name__ == "__main__":
    app.run(debug=True)
