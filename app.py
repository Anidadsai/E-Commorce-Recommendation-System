import streamlit as st
import pandas as pd
import random
import sklearn as sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load files
trending_products = pd.read_csv("trending_products.csv")
train_data = pd.read_csv("clean_data.csv")

# List of predefined image URLs
random_image_urls = [
    "static/img/img_1.png",
    "static/img/img_2.png",
    "static/img/img_3.png",
    "static/img/img_4.png",
    "static/img/img_5.png",
    "static/img/img_6.png",
    "static/img/img_7.png",
    "static/img/img_8.png",
]

# Function to truncate product name
def truncate(text, length=20):
    return text[:length] + "..." if len(text) > length else text

# Function for content-based recommendations
def content_based_recommendations(train_data, item_name, top_n=5):
    if item_name not in train_data['Name'].values:
        st.warning(f"Item '{item_name}' not found in the dataset.")
        return pd.DataFrame()

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix_content = tfidf_vectorizer.fit_transform(train_data['Tags'])
    cosine_similarities_content = cosine_similarity(tfidf_matrix_content, tfidf_matrix_content)

    item_index = train_data[train_data['Name'] == item_name].index[0]
    similar_items = list(enumerate(cosine_similarities_content[item_index]))
    similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)[1:top_n+1]

    recommended_item_indices = [x[0] for x in similar_items]
    return train_data.iloc[recommended_item_indices][['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']]

# Streamlit UI
st.title("E-commerce Product Recommendation System")

# Sidebar for product selection
st.sidebar.header("Product Selection")
selected_product = st.sidebar.selectbox("Choose a product:", train_data['Name'].unique())
top_n = st.sidebar.slider("Number of recommendations:", 1, 10, 5)

# Show trending products
st.subheader("Trending Products")
cols = st.columns(4)
for i, product in trending_products.head(4).iterrows():
    with cols[i % 4]:
        st.image(product["ImageURL"], width=150)
        st.write(truncate(product["Name"]))
        st.write(f"Brand: {product['Brand']}")
        st.write(f"Rating: {product['Rating']} ⭐")

# Get recommendations
if st.sidebar.button("Get Recommendations"):
    st.subheader(f"Recommended Products for: {selected_product}")
    recommendations = content_based_recommendations(train_data, selected_product, top_n)
    
    if not recommendations.empty:
        cols = st.columns(3)
        for i, product in recommendations.iterrows():
            with cols[i % 3]:
                st.image(product["ImageURL"], width=150)
                st.write(truncate(product["Name"]))
                st.write(f"Brand: {product['Brand']}")
                st.write(f"Rating: {product['Rating']} ⭐")
    else:
        st.error("No recommendations found.")

