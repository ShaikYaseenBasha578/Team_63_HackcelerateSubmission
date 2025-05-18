import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from utils import preprocess_user_input, filter_cars
from urllib.parse import quote
import base64

st.set_page_config(page_title="Car Recommender", layout="wide")

def get_base64_image(path):
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

logo_base64 = get_base64_image("images/logo.png")
background_image_url = "https://images.unsplash.com/photo-1553440569-bcc63803a83d?w=900&auto=format&fit=crop&q=60"

# --- Apply Dark Theme Styling ---
st.markdown(
    f"""
    <style>
     @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    
    .stApp {{
        background: linear-gradient(rgba(0,0,0,0.55), rgba(0,0,0,0.55)), url("{background_image_url}") !important;
        background-size: cover !important;
        background-position: center !important;
        background-repeat: no-repeat !important;
        font-family: 'Roboto', sans-serif;
        color: white;
    }}

    .navbar {{
        background-color: rgba(0, 0, 0, 0.6);
        backdrop-filter: blur(8px);
        padding: 1rem 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 2px solid #ffffff33;
        position: sticky;
        top: 0;
        z-index: 999;
    }}

    .navbar-logo img {{
        height: 200px;
        width: auto;
    }}

    .navbar-links a {{
        color: white;
        margin-left: 1.8rem;
        text-decoration: none;
        font-weight: 400;
        font-size: 1.8rem;
        transition: all 0.3s ease-in-out;
    }}

    .navbar-links a:hover {{
        text-decoration: underline;
        color: #ffcc00;
    }}


    .stTextInput, .stTextInput input {{
        background-color: #222 !important;
        color: white !important;
        border-radius: 10px;
        padding: 14px;
        font-size: 1.2rem;
    }}
    .stTextInput input:focus {{
        box-shadow: 0px 0px 10px rgba(255, 255, 255, 0.5);
    }}

    .search-btn {{
        background-color: #ffcc00;
        color: black;
        border: none;
        padding: 12px 26px;
        font-size: 1rem;
        font-weight: 500;
        border-radius: 8px;
        cursor: pointer;
        margin-top: 12px;
        transition: all 0.3s ease;
    }}

    .search-btn:hover {{
        background-color: #ffdb4d;
        box-shadow: 0 0 8px rgba(255, 204, 0, 0.4);
    }}

    .about-section {{
        padding: 4rem 2rem;
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        margin-top: 4rem;
        text-align: center;
        color: white;
    }}

    .about-section h2 {{
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }}

    .about-section p {{
        font-size: 1.1rem;
        max-width: 800px;
        margin: 0 auto 2.5rem auto;
        line-height: 1.6;
    }}

    .about-features {{
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 2rem;
    }}

    .feature {{
        background-color: rgba(0, 0, 0, 0.4);
        border-radius: 12px;
        padding: 2rem;
        max-width: 300px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }}

    .feature img {{
        height: 60px;
        margin-bottom: 1rem;
    }}

    .feature h4 {{
        font-size: 1.3rem;
        margin-bottom: 0.5rem;
    }}

    .feature p {{
        font-size: 1rem;
        line-height: 1.5;
    }}

    </style>
     <div class="navbar">
        <div class="navbar-logo">
             <img src="data:image/png;base64,{logo_base64}" alt="Logo">
        </div>
        <div class="navbar-links">
            <a href="#home">Home</a>
            <a href="#recommendations">Recommendations</a>
            <a href="#about">About</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Load Model and Data ---

@st.cache_resource
def load_assets():
    with open('car_recommendation_assets/label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)

    with open('car_recommendation_assets/categorical_mappings.pkl', 'rb') as f:
        categorical_mappings = pickle.load(f)

    return label_encoders, categorical_mappings

@st.cache_data
def load_data():
    return pd.read_csv("car_recommendation_assets/processed_dataset.csv")

label_encoders, categorical_mappings = load_assets()
df = load_data()

# --- User Input ---
st.markdown("##  All the Cars. One AI-Powered Search.")
st.markdown("Find your Dream Car")
user_query = st.text_input(
    "Describe what you're looking for",
    placeholder="e.g., 'Automatic SUV under 15 lakhs with cruise control'",
    help="Type your query to find the best car recommendations!"
)

st.markdown('<button class="search-btn">Search</button>', unsafe_allow_html=True)

# --- Run Recommendation ---
if user_query:
    try:
        # ✅ Extract filters
        filters = preprocess_user_input(user_query)

        # ✅ Ensure necessary mappings
        required_keys = [
            'make_mapping', 'model_mapping', 'variant_mapping',
            'reverse_make_mapping', 'reverse_model_mapping', 'reverse_variant_mapping'
        ]
        for key in required_keys:
            if key not in categorical_mappings:
                raise KeyError(f"Missing '{key}' in categorical_mappings.")

        make_mapping = categorical_mappings['make_mapping']
        model_mapping = categorical_mappings['model_mapping']
        variant_mapping = categorical_mappings['variant_mapping']
        reverse_make_mapping = categorical_mappings['reverse_make_mapping']
        reverse_model_mapping = categorical_mappings['reverse_model_mapping']
        reverse_variant_mapping = categorical_mappings['reverse_variant_mapping']

        # ✅ Filter dataset with proper arguments
        filtered_df = filter_cars(
            df, 
            filters,
            make_mapping=make_mapping,
            reverse_make_mapping=reverse_make_mapping,
            reverse_model_mapping=reverse_model_mapping,
            reverse_variant_mapping=reverse_variant_mapping
        )

        if filtered_df.empty:
            st.warning("No cars found matching your query.")
        else:
            # ✅ Normalize numerical features
            numerical_cols = ['Ex-Showroom_Price', 'Displacement', 'Power', 'ARAI_Certified_Mileage']
            scaler = StandardScaler()
            filtered_df[numerical_cols] = scaler.fit_transform(filtered_df[numerical_cols])

            # ✅ Extract categorical one-hot encoding
            encoded_cats = filtered_df.filter(like='_').values
            final_features = np.hstack((filtered_df[numerical_cols], encoded_cats))

            # ✅ Train KNN on filtered dataset
            knn_model = NearestNeighbors(n_neighbors=min(10, len(filtered_df)), metric='euclidean')
            knn_model.fit(final_features)

            # ✅ Convert query to feature vector
            query_vector = np.zeros(len(numerical_cols))
            query_categorical = np.zeros(encoded_cats.shape[1])
            one_hot_columns = list(df.filter(like='_').columns)

            for col in filters:
                if col in numerical_cols:
                    query_vector[numerical_cols.index(col)] = np.mean(filters[col]) if isinstance(filters[col], list) else filters[col]
                elif col in df.columns:
                    cat_column_name = col + "_" + str(filters[col])
                    if cat_column_name in one_hot_columns:
                        cat_index = one_hot_columns.index(cat_column_name)
                        query_categorical[cat_index] = 1

            full_query_vector = np.hstack((query_vector, query_categorical))

            # ✅ Get recommendations
        with st.spinner("Finding the best car recommendations..."):
            distances, indices = knn_model.kneighbors(full_query_vector.reshape(1, -1))

        

            # ✅ Display Results
            st.markdown("<h2 style='color: white;'>Top Car Recommendations:</h2>", unsafe_allow_html=True)
            sort_option = st.selectbox(
                "🔽 Sort By",
                ["None", "Price: Low to High", "Price: High to Low", "Mileage: High to Low", "Power: High to Low"],
                key="sort_selectbox"
            )

            
            selected_cars = []
            make_count = {}
            total_printed = 0
            max_recommendations = 5
            max_per_make = 2
            
            for index, distance in zip(indices[0], distances[0]):
                if total_printed >= max_recommendations:
                    break  
                
                car = filtered_df.iloc[index]
                make_name = car['Make']
                
                if make_count.get(make_name, 0) >= max_per_make:
                    continue  
                
                make_count[make_name] = make_count.get(make_name, 0) + 1
                selected_cars.append(car)
                total_printed += 1  
            

            make = car['Make'].replace(" ", "")
            model = car['Model'].replace(" ", "")
            variant = car['Variant'].replace(" ", "")

            # Sort logic
            if sort_option == "Price: Low to High":
                selected_cars = sorted(selected_cars, key=lambda x: x['Ex-Showroom_Price'])
            elif sort_option == "Price: High to Low":
                selected_cars = sorted(selected_cars, key=lambda x: x['Ex-Showroom_Price'], reverse=True)
            elif sort_option == "Mileage: High to Low":
                selected_cars = sorted(selected_cars, key=lambda x: x['ARAI_Certified_Mileage'], reverse=True)
            elif sort_option == "Power: High to Low":
                selected_cars = sorted(selected_cars, key=lambda x: x['Power'], reverse=True)

            for car in selected_cars:
                def format_cardekho_url(make, model):
                    make_slug = make.lower().replace(" ", "-")
                    model_slug = model.lower().replace(" ", "-")
                    return f"https://www.cardekho.com/{make_slug}/{model_slug}"

                def format_carwale_url(make, model):
                    make_slug = make.lower().replace(" ", "-")
                    model_slug = model.lower().replace(" ", "-")
                    variant_slug = variant.lower().replace(" ", "-")
                    return f"https://www.carwale.com/{make_slug}-cars/{model_slug}"
                 
                cardekho_link = format_cardekho_url(car['Make'], car['Model'])
                carwale_link = format_carwale_url(car['Make'], car['Model'])
                
                price = car['Ex-Showroom_Price'] * scaler.scale_[0] + scaler.mean_[0]
                mileage = car['ARAI_Certified_Mileage'] * scaler.scale_[3] + scaler.mean_[3]
                displacement = car['Displacement'] * scaler.scale_[1] + scaler.mean_[1]
                power = car['Power'] * scaler.scale_[2] + scaler.mean_[2]
                # ✅ Extract Transmission from one-hot columns
                transmission_type = 'Not specified'
                for col in car.index:
                    if col.startswith('Transmission_') and car[col] == 1:
                        transmission_type = col.replace('Transmission_', '')
                        transmission_type = transmission_type.replace('_', ' ').title()
                        break

                # ✅ Extract Fuel type from one-hot columns
                fuel_type = 'Not specified'
                for col in car.index:
                    if col.startswith('Fuel_Type_') and car[col] == 1:
                        fuel_type = col.replace('Fuel_Type_', '')
                        fuel_type = fuel_type.replace('_', ' ').title()
                        break
                
                
                
                st.markdown(f"""
                <span style='font-size:18px; font-weight:bold; color:white;'>
                 {car['Make']} {car['Model']} {car['Variant']} – ₹{price:,.0f}
                </span>
                """, unsafe_allow_html=True)

                with st.expander("View Details"):
                    st.markdown(f"""
                    <div style="
                        background-color: #1e1e1e; 
                        padding: 16px 20px; 
                        border-radius: 12px; 
                        box-shadow: 0 0 12px rgba(0,0,0,0.4); 
                        font-family: 'Segoe UI', sans-serif;
                        color: #f0f0f0;
                        margin-top: 10px;
                    ">
                        <p style="margin: 6px 0;">💰 <strong style="color:#ffcc00;">Price:</strong> ₹{price:,.0f}</p>
                        <p style="margin: 6px 0;">⛽ <strong style="color:#ffcc00;">Mileage:</strong> {mileage:.1f} kmpl</p>
                        <p style="margin: 6px 0;">⚙️ <strong style="color:#ffcc00;">Engine:</strong> {displacement:.1f} cc</p>
                        <p style="margin: 6px 0;">🌀 <strong style="color:#ffcc00;">Power:</strong> {power:.0f} BHP</p>
                        <p style="margin: 6px 0;">🧰 <strong style="color:#ffcc00;">Transmission:</strong> {transmission_type}</p>
                        <p style="margin: 6px 0;">🧰 <strong style="color:#ffcc00;">Fuel:</strong> {fuel_type}</p>
                        <p style="margin: 6px 0;">👥 <strong style="color:#ffcc00;">Seating:</strong> {int(car.get('Seating_Capacity', 0))} Seater</p>
                        <p>
                        <div style="margin-top: 10px;">
                            <a href="{cardekho_link}" target="_blank" 
                                style="color:#add8e6; text-decoration:none; margin-right:15px;">
                                🔗 View on CarDekho
                            </a>
                            <a href="{carwale_link}" target="_blank" 
                                style="color:#add8e6; text-decoration:none;">
                                🔗 View on CarWale
                            </a>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ✅ Optional direct comparison with similar cars
                    st.markdown("**🔍 Similar cars from same Make:**")
                    similar_cars_encoded = filtered_df[
                        (filtered_df['Make'] == car['Make']) &
                        (filtered_df['Model'] != car['Model']) &
                        (filtered_df['Variant'] != car['Variant'])
                    ].head(3)

                    if similar_cars_encoded.empty:
                        st.write("No similar cars found.")
                    else:
                        for _, sim_car in similar_cars_encoded.iterrows():
                            sim_make = sim_car['Make']
                            sim_model = sim_car['Model']
                            sim_variant = sim_car['Variant']
                            sim_price = sim_car['Ex-Showroom_Price'] * scaler.scale_[0] + scaler.mean_[0]

                            st.markdown(f"- {sim_make} {sim_model} {sim_variant} – ₹{sim_price:,.0f}")

            # ✅ Mark top recommendations in the DataFrame
            top_indices = [car.name for car in selected_cars]
            styled_df = filtered_df.copy()
            styled_df['🏆 Recommended'] = pd.Series(
                styled_df.index.isin(top_indices), index=styled_df.index
            ).map({True: "✅", False: ""})

            # ✅ Move the "Recommended" column to the front
            columns = ['🏆 Recommended'] + [col for col in styled_df.columns if col != '🏆 Recommended']
            styled_df = styled_df[columns]

            # ✅ Style the DataFrame
            def highlight_recommended(row):
                return ['background-color: #1f4f24; color: white; font-weight: bold;' if row['🏆 Recommended'] == '✅' else '' for _ in row]

            styled = styled_df.style.apply(highlight_recommended, axis=1)\
                .set_table_styles([
                    {'selector': 'th', 'props': [('background-color', '#333'), ('color', 'white'), ('font-size', '14px')]},
                    {'selector': 'td', 'props': [('background-color', '#222'), ('color', 'white')]},
                    {'selector': 'tr:hover td', 'props': [('background-color', '#444 !important')]}
                ])

            with st.expander("📋 All the cars that fit your criteria"):
                st.dataframe(styled, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error: {e}")

    # --- ABOUT SECTION ---
st.markdown(f"""
<div id="about" class="about-section">
    <h2>How CarRec Works</h2>
    <p>We use advanced AI models to help you find the perfect car - from all brand-new models. We analyze your needs and filter listings from multiple sources in a single search.</p>
""", unsafe_allow_html=True)

html_code = """
<div class="about-features">
    <div class="feature">
        <img src="https://img.icons8.com/?size=100&id=fsoiqMUp0O4v&format=png&color=000000" alt="Filters">
        <h4>Just the Cars You Want</h4>
        <p>Use powerful filters - fuel type, body style, budget, and more - to find exactly what you're looking for.</p>
    </div>
    <div class="feature">
        <img src="https://img.icons8.com/?size=100&id=97624&format=png&color=000000" alt="Smart Results">
        <h4>Smart Results</h4>
        <p>We recommend cars that match your preferences using intelligent KNN-based algorithms for accurate results.</p>
    </div>
    <div class="feature">
        <img src="https://img.icons8.com/?size=100&id=97108&format=png&color=000000" alt="Device Compatibility">
        <h4>Search on Any Device</h4>
        <p>Whether you're on your phone, tablet, or laptop, our app works seamlessly on any screen size.</p>
    </div>
</div>
"""

st.markdown(html_code, unsafe_allow_html=True) 

st.markdown("""
    <hr style="margin-top: 2em; border: 1px solid #444;">
    <div style="text-align: center; color: grey; font-size: 0.9em;">
        Built by <strong>Shaik Yaseen Basha</strong><br>
        🔗 <a href="https://github.com/ShaikYaseenBasha578" style="color:#add8e6;" target="_blank">GitHub</a> |
        <a href="https://www.linkedin.com/in/shaik-yaseen-basha-a90627287/" style="color:#add8e6;" target="_blank">LinkedIn</a><br>
        📊 Data sourced from CarDekho and CarWale
    </div>
""", unsafe_allow_html=True)

