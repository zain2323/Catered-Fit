import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

df = pd.read_csv("./data/updated_foodsnew2.csv")

def unique_ingredients(df):
    new_ingredients = []
    ingredients = df["ingredients"]
    for ingredient in ingredients:
        lst = ingredient.split(",")
        lst = list(map(str.lower, lst))
        for i in lst:
            if i not in new_ingredients and i != " ":
                new_ingredients.append(i.strip())
    new_ingredients = sorted(set(new_ingredients))
    return new_ingredients

def unique_course(df):
    return df["course"].unique()
    
def hot_encoding(df, courses, ingredients, ocassions):
    dataframe = df.copy()
    for course in courses:
        dataframe[course] = 0
        dataframe.loc[dataframe.course.str.contains(course), [course]] = 1
    for i in ingredients:
        dataframe[i] = 0
        dataframe.loc[dataframe.ingredients.str.contains(i), [i]] = 1
    for ocassion in ocassions:
        dataframe[ocassion] = 0
        dataframe.loc[dataframe.ocassion.str.contains(ocassion), [ocassion]] = 1
    # Giving 'any' column value of 1
    df["any"] = 1
    return dataframe

def encode_course(df, course, courses):
    dataframe = df.copy()
    for course in courses:
        dataframe[course] = 0
        dataframe.loc[dataframe.course.str.contains(course), [course]] = 1
    return dataframe

def remove_columns(df):
    columns = ["name", "ingredients", "diet", "flavor_profile", "course", "images", "ocassion", "prep_time", "cook_time"]
    return df.drop(columns, axis=1)

def get_encoded_label(df):
    dataframe = df.copy()
    label = dataframe["name"]
    le = preprocessing.LabelEncoder()
    label_encoded = le.fit_transform(label)
    dataframe["name_encoded"] = label_encoded
    encoded = dict(zip(le.classes_, range(len(le.classes_))))
    return dataframe, encoded 

def split_dataset(dataframe):
  X = dataframe[dataframe.columns[1:]].values
  y = dataframe[dataframe.columns[0]].values
  return X, y

def transform_input(input_ing, input_course, input_ocassion, courses, ingredients):
    vec = []
    for course in courses:
        if course in input_course:
            vec.append(1)
        else:
            vec.append(0)
    for ingredient in ingredients:
        if ingredient in input_ing:
            vec.append(1)
        else:
            vec.append(0)
    for ocassion in ["eid_fitr", "eid_adha", "new_year", "any"]:
        if ocassion in input_ocassion:
            vec.append(1)
        else:
            vec.append(0)  
    vec = np.array(vec).reshape(1,-1)
    return vec

def getFoodName(foodIdList, X_dict):
    food = []
    for id in foodIdList:
        for key, value in X_dict.items():
            if value == id:
                food.append(key)
    return food

def get_food_from_idx(df, index):
    name = df[df.index == index]["name"].values[0]
    image = df[df.index == index]["images"].values[0]
    return (name, image)

def get_images_from_id(id_list):
    images = []
    for id in id_list:
        images.append(df[df.index == id]["images"].values[0])
    return images

def get_food_idx_list(name_list):
    ids = []
    for food in name_list:
        cnt = 0
        for name in df["name"]:
            if food == name:
                ids.append(cnt)
            cnt += 1 
    return ids

def get_food_idx(df, food):
    cnt = 0
    for name in df["name"]:
        if food == name:
            return cnt
        cnt += 1 
    return -1

def get_similar_foods(name):
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(df["ingredients"])
    cosine_sim = cosine_similarity(count_matrix)
    id = get_food_idx(df, name)
    foods = []
    images = []
    similar_foods = list(enumerate(cosine_sim[id]))
    similar_foods = sorted(similar_foods,  key=lambda x:x[1], reverse=True)
    i = 0
    for food in similar_foods:
        if i > 8:
            break
        food, image = get_food_from_idx(df, food[0])
        foods.append(food)
        images.append(image)
        i += 1
    return (foods[1:], images[1:])

def initialize():
    ingredients = unique_ingredients(df)
    courses = unique_course(df)
    ocassions = ["eid_fitr", "eid_adha", "new_year", "any"]
    dataframe, X_dict = get_encoded_label(df)
    dataframe = hot_encoding(dataframe, courses, ingredients, ocassions)
    dataframe = remove_columns(dataframe)
    X, y = split_dataset(dataframe)
    return (ingredients, courses, X_dict, X, y)


def getFoodDetails(name):
    return df[df.name == name]

def train(X, y):
    # Fitting the model
    rfc = RandomForestClassifier(n_jobs=-1) 
    rfc.fit(X,y)

    # Saving the model
    pickle.dump(rfc, open('trained-model.pkl','wb'))


try:
    # Loading the model
    model = pickle.load(open('trained-model.pkl','rb'))
except:
    ingredients, courses, X_dict, X, y = initialize()
    train(X, y)