import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random

# Load the data
data = pd.read_csv("D:/Codes/minorproject/data/food_suggestions (2).csv")
nutrition_data = pd.read_csv("D:/Codes/minorproject/data/nutrition_distriution.csv")

Food_itemsdata = data['Name']

def process_data_and_predict(age, weight, height, target):
    # Calculate BMI
    bmi = weight / ((height / 100) ** 2)
    agecl = age // 20

    st.write(f"Your body mass index is: {bmi}")
    if bmi < 16:
        st.write("According to your BMI, you are Severely Underweight")
        clbmi = 4
    elif bmi >= 16 and bmi < 18.5:
        st.write("According to your BMI, you are Underweight")
        clbmi = 3
    elif bmi >= 18.5 and bmi < 25:
        st.write("According to your BMI, you are Healthy")
        clbmi = 2
    elif bmi >= 25 and bmi < 30:
        st.write("According to your BMI, you are Overweight")
        clbmi = 1
    else:
        st.write("According to your BMI, you are Severely Overweight")
        clbmi = 0

    ti = (clbmi + agecl) / 2

    # Prepare nutrition distribution data
    weight_gain_cat = nutrition_data.iloc[[0, 1, 2, 3, 4, 7, 9, 10]].T
    weight_loss_cat = nutrition_data.iloc[[1, 2, 3, 4, 6, 7, 9]].T

    weight_gain_data = weight_gain_cat.to_numpy()
    weight_loss_data = weight_loss_cat.to_numpy()

    # Generate training data for each category
    def prepare_training_data(category_data):
        fin_data = []
        for zz in range(5):
            for jj in range(len(category_data)):
                valloc = list(category_data[jj])
                valloc.append(bmicls[zz])
                valloc.append(agecls[zz])
                fin_data.append(np.array(valloc))
        return np.array(fin_data)

    bmicls = [0, 1, 2, 3, 4]
    agecls = [0, 1, 2, 3, 4]

    weight_gain_fin = prepare_training_data(weight_gain_data)
    weight_loss_fin = prepare_training_data(weight_loss_data)

    # Select data based on target goal
    if target == 'weight_gain':
        X_test = weight_gain_data
        X_train = weight_gain_fin
    elif target == 'weight_loss':
        X_test = weight_loss_data
        X_train = weight_loss_fin
    else:
        st.write("Invalid target specified. Please enter 'weight_gain' or 'weight_loss'.")
        return

    # Align X_test with X_train by adding bmi and age categories
    X_test_aligned = []
    for zz in range(5):
        for row in X_test:
            valloc = list(row)
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            X_test_aligned.append(np.array(valloc))

    X_test_aligned = np.array(X_test_aligned)

    # Split the data for accuracy testing
    X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_train, [0] * len(X_train), test_size=0.2, random_state=42)

    # Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train_split, y_train_split)
    y_pred_split = clf.predict(X_test_split)

    # Calculate Random Forest accuracy
    rf_accuracy = accuracy_score(y_test_split, y_pred_split) * 100

    st.write(f"Random Forest accuracy: {rf_accuracy:.2f}%")

    # Predict on the actual test data for recommendation
    y_pred = clf.predict(X_test_aligned)

    st.write('SUGGESTED FOOD ITEMS ::')
    recommended_items = []
    for ii in range(len(y_pred)):
        if y_pred[ii] == 0:
            findata = Food_itemsdata[ii % len(Food_itemsdata)]
            recommended_items.append(findata)

    # Filter recommendations based on nutritional content
    def filter_recommendations(items, bmi_category):
        filtered_items = []
        for item in items:
            item_data = data[data['Name'] == item]
            fats = item_data['fats'].values[0]
            carbohydrates = item_data['carbohydrates'].values[0]
            fiber = item_data['fiber'].values[0]
            protein = item_data['protein'].values[0]
            sugar = item_data['sugar'].values[0]

            # Logic to filter based on nutritional needs (example logic)
            if bmi_category == 4 or bmi_category == 3:  # Underweight categories
                if protein > 10 and fats > 5 and sugar < 15:
                    filtered_items.append(item)
            elif bmi_category == 2:  # Healthy category
                if fiber > 5 and protein > 5 and fats < 15:
                    filtered_items.append(item)
            elif bmi_category == 1 or bmi_category == 0:  # Overweight categories
                if fiber > 10 and fats < 10 and sugar < 10:
                    filtered_items.append(item)

        return filtered_items

    filtered_items = filter_recommendations(recommended_items, clbmi)

    # Shuffle and select a random subset of filtered items
    random.shuffle(filtered_items)
    display_items = filtered_items[:10]  # Display up to 10 items

    for item in display_items:
        item_data = data[data['Name'] == item]
        st.write(f"{item}:")
        st.write(f"fats: {item_data['fats'].values[0]} g")
        st.write(f"carbohydrates: {item_data['carbohydrates'].values[0]} g")
        st.write(f"fiber: {item_data['fiber'].values[0]} g")
        st.write(f"protein: {item_data['protein'].values[0]} g")
        st.write(f"sugar: {item_data['sugar'].values[0]} g")

    st.write('\nThank You for taking our recommendations. :)')

def main():
    st.title("Food Recommendation System")
    st.write("Please provide some information to receive personalized food recommendations.")

    # Get user input
    age = st.number_input("Enter Age", min_value=1, max_value=120, step=1)
    weight = st.number_input("Enter Weight (kg)", min_value=1.0, max_value=500.0, step=0.1)
    height = st.number_input("Enter Height (cm)", min_value=1.0, max_value=300.0, step=0.1)
    target = st.selectbox("Select Target Goal", ('Weight Gain', 'Weight Loss'))

    if st.button("Get Recommendations"):
        # Call the function to process data and make predictions
        target = target.lower().replace(" ", "_")
        process_data_and_predict(age, weight, height, target)

if __name__ == "__main__":
    main()
