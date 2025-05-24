import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import plotly.express as px
import joblib
import base64
from io import StringIO
import os


st.set_page_config(
    page_title="Molecular Biology Data Analysis",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)



# sidebar navigation
st.sidebar.title("🧬 Molecular Biology Analysis")
st.sidebar.markdown("---")
app_mode = st.sidebar.selectbox(
    "Choose the app mode",
    ["Home", "1. Data Upload", "2. K-Means Clustering", "3. Adjustable Clustering", 
     "4. Interactive Plotting", "5. Multi-tab Interface", "6. Sidebar Settings", 
     "7. Dynamic Filtering", "8. Train Classification Model", "9. Prediction", "10. About the Project Team"]
)

# data downloader utility function
def get_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# saving models utility
def get_model_download_link(model, filename, text):
    """Generate a link to download the model"""
    buffer = StringIO()
    joblib.dump(model, filename)
    with open(filename, 'rb') as f:
        model_data = f.read()
    os.remove(filename)  # Clean up the file
    b64 = base64.b64encode(model_data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Home page
if app_mode == "Home":
    # background image
    def add_bg_from_url():
        # urls for images
        light_image_url = "https://www.ppt-backgrounds.net/thumbs/dna-design-downloads.jpeg"
        dark_image_url = "https://www.ppt-backgrounds.net/thumbs/dna-design-downloads.jpeg"
        
        return f"""
        <style>
        /* Light mode - target the specific Streamlit container */
        [data-testid="stAppViewContainer"] {{
            background: url("{light_image_url}");
            background-size: cover;
            background-position: center center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        /* Dark mode - override when dark theme is active */
        [data-theme="dark"] [data-testid="stAppViewContainer"] {{
            background: url("{dark_image_url}");
            background-size: cover;
            background-position: center center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        
        /* Make content area transparent */
        [data-testid="stHeader"] {{
            background-color: rgba(0,0,0,0);
        }}
        
        [data-testid="stToolbar"] {{
            right: 2rem;
        }}
        </style>
        """
    
    st.markdown(add_bg_from_url(), unsafe_allow_html=True)
    
    st.title("Molecular Biology Data Analysis Web App")
    st.write("""
    ### Welcome to the Molecular Biology Data Analysis Web Application!
    
    This application provides tools for analyzing molecular biology data through various modules:
    
    1. **Data Upload**: Import CSV files for analysis
    2. **K-Means Clustering**: Basic clustering of your data
    3. **Adjustable Clustering**: K-Means with customizable parameters
    4. **Interactive Plotting**: Create custom scatter plots
    5. **Multi-tab Interface**: Explore data through different views
    6. **Sidebar Settings**: Configure your analysis with a sidebar
    7. **Dynamic Filtering**: Filter your data based on text input
    8. **Train Classification Model**: Build a Random Forest classifier
    9. **Prediction**: Use trained models to make predictions
    10. **About the Project Team**: Information about the creators of this web app
    
    To begin, select an option from the sidebar and upload your data.
    """)


    
    st.info("This application is designed for molecular biology data analysis but can be used with any tabular data.")
    
    # Sample dataset option
    if st.button("Use Sample Dataset"):
        # create example data
        np.random.seed(42)
        n_samples = 100
        
        # gene expression data (4 genes)
        gene_expr = np.random.normal(0, 1, size=(n_samples, 4))
        gene_cols = [f"Gene_{i}" for i in range(1, 5)]
        
        # create labels for the last column
        species = np.random.choice(['setosa', 'versicolor', 'virginica'], size=n_samples)
        
        # create dataframe
        example_data = pd.DataFrame(gene_expr, columns=gene_cols)
        example_data['Species'] = species
        
        # save to CSV
        example_data.to_csv("sample_data.csv", index=False)
        st.success("Sample dataset created! You can now use it in any module.")

# 1. Data Upload
elif app_mode == "1. Data Upload":
    st.title("1. Basic Data Uploader")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data:")
        st.dataframe(df)
        
        # Add download option
        st.markdown(
            get_download_link(df, "uploaded_data.csv", "Download Data"),
            unsafe_allow_html=True
        )

# 2. K-Means Clustering
elif app_mode == "2. K-Means Clustering":
    st.title("2. Data Uploader + K-Means Clustering")
    
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data:")
        st.dataframe(df)
        
        # Perform K-Means clustering
        try:
            kmeans = KMeans(n_clusters=3, random_state=42)
            df['Cluster'] = kmeans.fit_predict(df.iloc[:, :-1])
            
            st.write("### Clustering Results:")
            st.dataframe(df)
            
            # Add download option
            st.markdown(
                get_download_link(df, "clustering_results.csv", "Download Results"),
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"Error during clustering: {e}")
            st.info("Make sure your data is suitable for clustering. All columns except the last should be numeric.")

# 3. Adjustable Clustering
elif app_mode == "3. Adjustable Clustering":
    st.title("3. K-Means Clustering with Adjustable K")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data:")
        st.dataframe(df)
        
        k = st.slider("Select number of clusters", 2, 10, 3)
        
        try:
            kmeans = KMeans(n_clusters=k, random_state=42)
            df['Cluster'] = kmeans.fit_predict(df.iloc[:, :-1])
            
            st.write("### Clustering Results:")
            st.dataframe(df)
            
            # Add visualization
            if len(df.columns) >= 3:  # Need at least 2 features + cluster column
                st.write("### Cluster Visualization")
                fig = px.scatter(
                    df, 
                    x=df.columns[0], 
                    y=df.columns[1], 
                    color='Cluster',
                    title=f"Clusters with k={k}"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Add download option
            st.markdown(
                get_download_link(df, "clustering_results.csv", "Download Results"),
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"Error during clustering: {e}")
            st.info("Make sure your data is suitable for clustering. All columns except the last should be numeric.")

# 4. Interactive Plotting
elif app_mode == "4. Interactive Plotting":
    st.title("4. Interactive Scatter Plot")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data:")
        st.dataframe(df)
        
        # Select X and Y axes
        x_axis = st.selectbox("Select X-axis", df.columns[:-1])
        y_axis = st.selectbox("Select Y-axis", df.columns[:-1])

        # Create scatter plot
        fig = px.scatter(df, x=x_axis, y=y_axis, color=df.columns[-1])
        st.plotly_chart(fig)
        
        # Add option to download the plot
        st.write("### Download Options")
        st.write("You can download the plot by clicking the camera icon in the top-right corner of the plot.")

# 5. Multi-tab Interface
elif app_mode == "5. Multi-tab Interface":
    st.title("5. Multi-tab Interface")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        tab1, tab2, tab3 = st.tabs(["Dataset", "Statistics", "Plots"])
        with tab1:
            st.write("### Uploaded Data")
            st.dataframe(df)
        with tab2:
            st.write("### Summary Statistics")
            st.write(df.describe())
        with tab3:
            x_axis = st.selectbox("X-axis", df.columns[:-1])
            y_axis = st.selectbox("Y-axis", df.columns[:-1])
            fig = px.scatter(df, x=x_axis, y=y_axis, color=df.columns[-1])
            st.plotly_chart(fig)

# 6. Sidebar Settings
elif app_mode == "6. Sidebar Settings":
    st.title("6. Sidebar for Settings")
    st.sidebar.header("Settings")
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data")
        st.dataframe(df)
        
        # Add clustering option
        if st.sidebar.checkbox("Perform Clustering"):
            k = st.sidebar.slider("Number of Clusters", 2, 10, 3)
            kmeans = KMeans(n_clusters=k, random_state=42)
            df['Cluster'] = kmeans.fit_predict(df.iloc[:, :-1])
            st.write("### Clustering Results")
            st.dataframe(df)
            
            # Add visualization
            if st.sidebar.checkbox("Show Visualization"):
                fig = px.scatter(
                    df, 
                    x=df.columns[0], 
                    y=df.columns[1], 
                    color='Cluster',
                    title=f"Clusters with k={k}"
                )
                st.plotly_chart(fig, use_container_width=True)

# 7. Dynamic Filtering
elif app_mode == "7. Dynamic Filtering":
    st.title("7. Dynamic Filtering")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data")
        st.dataframe(df)
        
        # Add column selector
        filter_column = st.selectbox("Select column to filter", df.columns)
        search_term = st.text_input(f"Filter {filter_column}:")
        
        if search_term:
            filtered_df = df[df[filter_column].astype(str).str.contains(search_term, case=False, na=False)]
            st.write(f"### Filtered Results ({len(filtered_df)} rows)")
            st.dataframe(filtered_df)
            
            # Add download option for filtered data
            if len(filtered_df) > 0:
                st.markdown(
                    get_download_link(filtered_df, "filtered_data.csv", "Download Filtered Data"),
                    unsafe_allow_html=True
                )

# 8. Train Classification Model
elif app_mode == "8. Train Classification Model":
    st.title("8. Train a Classification Model")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data")
        st.dataframe(df)
        
        # Add model parameters
        st.write("### Model Parameters")
        test_size = st.slider("Test size (%)", 10, 50, 20) / 100
        n_estimators = st.slider("Number of trees", 10, 200, 100, 10)
        
        if st.button("Train Model"):
            try:
                X = df.iloc[:, :-1]
                y = df.iloc[:, -1]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                
                clf = RandomForestClassifier(n_estimators=n_estimators)
                clf.fit(X_train, y_train)
                preds = clf.predict(X_test)
                acc = accuracy_score(y_test, preds)
                
                st.write(f"### Model Accuracy: {acc:.2f}")
                
                # Save model
                model_filename = "random_forest_model.pkl"
                joblib.dump(clf, model_filename)
                
                # Provide download link
                with open(model_filename, 'rb') as f:
                    model_bytes = f.read()
                b64 = base64.b64encode(model_bytes).decode()
                href = f'<a href="data:application/octet-stream;base64,{b64}" download="{model_filename}">Download Trained Model</a>'
                st.markdown(href, unsafe_allow_html=True)
                
                # Clean up
                if os.path.exists(model_filename):
                    os.remove(model_filename)
                
                # Feature importance
                st.write("### Feature Importance")
                feature_importance = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': clf.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                st.dataframe(feature_importance)
                
                # Visualize feature importance
                fig = px.bar(
                    feature_importance,
                    x='Feature',
                    y='Importance',
                    title='Feature Importance'
                )
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error during model training: {e}")
                st.info("Make sure your data is suitable for classification. All columns except the last should be numeric, and the last column should contain the classes.")

# 9. Prediction
elif app_mode == "9. Prediction":
    st.title("9. Upload Test Data & Predict")
    
    uploaded_model = st.file_uploader("Upload Trained Model", type=["pkl"])
    uploaded_test = st.file_uploader("Upload Test CSV", type=["csv"])

    if uploaded_model and uploaded_test:
        try:
            # Load trained model
            model = joblib.load(uploaded_model)

            # Load test dataset
            test_data = pd.read_csv(uploaded_test)
            
            st.write("### Test Data")
            st.dataframe(test_data)

            # Check for feature_names_in_ attribute
            if hasattr(model, 'feature_names_in_'):
                expected_features = model.feature_names_in_
                
                # Check if all expected features are present
                missing_features = [f for f in expected_features if f not in test_data.columns]
                if missing_features:
                    st.error(f"Missing features in test data: {', '.join(missing_features)}")
                    st.stop()
                
                # Filter test data to include only expected features
                test_data_filtered = test_data[expected_features]
            else:
                # If model doesn't have feature_names_in_, use all columns except the last (if it exists)
                if len(test_data.columns) > 1:
                    test_data_filtered = test_data.iloc[:, :-1]
                else:
                    test_data_filtered = test_data
            
            if st.button("Make Predictions"):
                # Make predictions
                predictions = model.predict(test_data_filtered)

                # Add predictions to the dataframe
                result_data = test_data.copy()
                result_data["Predicted Class"] = predictions

                # Display results
                st.write("### Predictions")
                st.dataframe(result_data)
                
                # Add download option
                st.markdown(
                    get_download_link(result_data, "predictions.csv", "Download Predictions"),
                    unsafe_allow_html=True
                )
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    elif not uploaded_model:
        st.info("Please upload a trained model file (.pkl)")
    elif not uploaded_test:
        st.info("Please upload test data (.csv)")

# 10 Names and contributions
elif app_mode == "10. About the Project Team":
    st.title("About Us!")
    st.write("""
    ### A page about the team members and each one's contributions on the project.
    
    
    1.  **ΗΛΙΑΣ ΣΟΥΛΕΛΕΣ-inf2021208:** Python code writing, Docker config, testing and debugging
    2.  **ΕΛΕΥΘΕΡΙΟΣ-ΓΕΩΡΓΙΟΣ ΜΠΟΝΤΗΣ-inf2021160:** Latex writing, testing
    3.  **ΑΘΑΝΑΣΙΑΔΗΣ ΠΑΝΑΓΙΩΤΗΣ-Π2015167:** Python code writing, Docker config, GitHub repository owner, testing and debugging
    
        
    Thank you for using our app for the analysis of molecular biology data. We hope we helped you!
    """)