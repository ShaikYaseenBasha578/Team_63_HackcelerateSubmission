# 🚘 Car Recommendation System

This is a Streamlit-based car recommendation web app that allows users to input natural language queries like:

> *"Automatic SUV under 15 lakhs with cruise control"*

The app processes the input, generates vector representations, and recommends cars using a K-Nearest Neighbors (KNN) model. It also supports sorting by price, mileage, and power, and includes direct links to CarDekho and CarWale for more details.

---

## 🗂️ Project Structure

### `Car_Rec_Main.py`

This is the main file that runs the Streamlit app. It handles:

* UI layout and theme
* Text input from the user
* Loading spinners and car display logic
* Sorting functionality
* Links to external car sites

### `utils.py`

This file contains helper functions such as:

* `preprocess_user_input()`: Converts user query into a format the model understands
* `load_model_and_data()`: Loads the trained KNN model and car dataset
* `recommend_cars()`: Core logic for finding nearest cars based on vector similarity

Both files are **required** to run the app successfully.

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
streamlit run Car_Rec_Main.py
```

---

## 📅 Submitted on

**18 May 2025**

