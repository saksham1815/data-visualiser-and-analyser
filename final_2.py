# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle

# Scikit-learn imports for preprocessing, modeling and evaluation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
from sklearn.cluster import KMeans

# Set the page configuration
st.set_page_config(page_title="Data Analysis & ML Web App", layout="wide")


# =============================================================================
# Data Upload and Preprocessing Functions
# =============================================================================
def load_data(uploaded_file, encoding):
    try:
        df = pd.read_csv(uploaded_file, encoding=encoding)
        st.success("Data loaded successfully!")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def clean_data(df):
    st.write("**Cleaning Data:** Handling missing values and removing duplicates...")
    imputer_numeric = SimpleImputer(strategy="median")
    imputer_categorical = SimpleImputer(strategy="most_frequent")
    
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = imputer_numeric.fit_transform(df[[col]])
        else:
            df[col] = imputer_categorical.fit_transform(df[[col]])
    
    df = df.drop_duplicates()
    st.write("Data cleaning completed.")
    return df

def transform_data(df):
    st.write("**Transforming Data:** Applying normalization/standardization and encoding...")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    transformation_choice = st.selectbox("Select Transformation", ["None", "Normalization", "Standardization"], key="transform")
    
    if transformation_choice == "Normalization":
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        st.write("Data normalized using MinMaxScaler.")
    elif transformation_choice == "Standardization":
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        st.write("Data standardized using StandardScaler.")
    else:
        st.write("No transformation applied.")
    
    # Encoding categorical variables if any
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        st.write("Encoding categorical variables:", cat_cols)
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df

def handle_outliers(df):
    st.write("**Outlier Management:**")
    outlier_option = st.selectbox("Handle Outliers?", ["None", "Remove Outliers (IQR Method)", "Cap Outliers (IQR Method)"], key="outliers")
    if outlier_option == "None":
        st.write("No outlier treatment applied.")
        return df
    
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if not numeric_cols:
        st.warning("No numeric columns available for outlier treatment.")
        return df
    
    df_out = df.copy()
    for col in numeric_cols:
        Q1 = df_out[col].quantile(0.25)
        Q3 = df_out[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        if outlier_option == "Remove Outliers (IQR Method)":
            before = df_out.shape[0]
            df_out = df_out[(df_out[col] >= lower_bound) & (df_out[col] <= upper_bound)]
            after = df_out.shape[0]
            st.write(f"Removed {before - after} outlier rows from column '{col}'.")
        elif outlier_option == "Cap Outliers (IQR Method)":
            df_out[col] = np.where(df_out[col] < lower_bound, lower_bound, df_out[col])
            df_out[col] = np.where(df_out[col] > upper_bound, upper_bound, df_out[col])
            st.write(f"Capped outliers in column '{col}'.")
    return df_out

def reduce_data(df):
    st.write("**Data Reduction (Optional):**")
    reduction_option = st.selectbox("Select Reduction Technique", ["None", "Feature Selection", "Dimensionality Reduction (PCA)"], key="reduce")
    if reduction_option == "Feature Selection":
        selected_cols = df.columns[:len(df.columns)//2]
        st.write("Selected features:", list(selected_cols))
        return df[selected_cols]
    elif reduction_option == "Dimensionality Reduction (PCA)":
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if len(numeric_cols) < 2:
            st.warning("Not enough numeric features for PCA.")
            return df
        pca = PCA(n_components=2)
        components = pca.fit_transform(df[numeric_cols])
        pca_df = pd.DataFrame(components, columns=['PC1', 'PC2'])
        st.write("PCA applied. Explained Variance Ratio:", pca.explained_variance_ratio_)
        return pca_df
    else:
        return df

def preprocess_data(raw_df):
    """Run cleaning, transformation, outlier management, and reduction on the raw dataset."""
    df = clean_data(raw_df)
    df = transform_data(df)
    df = handle_outliers(df)
    df = reduce_data(df)
    return df


# =============================================================================
# Visualization Functions Using Plotly
# =============================================================================
def plot_univariate(df):
    st.subheader("Univariate Analysis")
    column = st.selectbox("Select column", df.columns, key="uni_col")
    plot_type = st.selectbox("Select plot type", ["Histogram", "Box Plot", "KDE-like Plot", "Bar Chart"], key="uni_plot")
    
    if plot_type == "Histogram":
        fig = px.histogram(df, x=column, nbins=30, title=f"Histogram of {column}")
    elif plot_type == "Box Plot":
        fig = px.box(df, y=column, title=f"Box Plot of {column}")
    elif plot_type == "KDE-like Plot":
        fig = px.histogram(df, x=column, nbins=30, title=f"KDE-like Plot of {column}",
                           marginal="violin", histnorm='density')
    elif plot_type == "Bar Chart":
        counts = df[column].value_counts().reset_index()
        counts.columns = [column, "count"]
        fig = px.bar(counts, x=column, y="count", title=f"Bar Chart of {column}")
    st.plotly_chart(fig, use_container_width=True)

def plot_bivariate(df):
    st.subheader("Bivariate Analysis")
    cols = st.multiselect("Select two columns", df.columns, max_selections=2, key="bi_cols")
    if len(cols) != 2:
        st.warning("Please select exactly two columns.")
        return
    plot_type = st.selectbox("Select plot type", ["Scatter Plot", "Line Plot"], key="bi_plot")
    if plot_type == "Scatter Plot":
        fig = px.scatter(df, x=cols[0], y=cols[1], title=f"Scatter Plot of {cols[0]} vs {cols[1]}")
    else:
        fig = px.line(df, x=cols[0], y=cols[1], title=f"Line Plot of {cols[0]} vs {cols[1]}")
    st.plotly_chart(fig, use_container_width=True)

def plot_multivariate(df):
    st.subheader("Multivariate Analysis")
    option = st.selectbox("Select plot", ["Scatter Matrix (Pair Plot)", "3D Scatter Plot", "Cluster Plot"], key="multi")
    if option == "Scatter Matrix (Pair Plot)":
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        if numeric_df.empty:
            st.warning("No numeric columns available.")
            return
        fig = px.scatter_matrix(numeric_df, title="Scatter Matrix")
        st.plotly_chart(fig, use_container_width=True)
    elif option == "3D Scatter Plot":
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if len(numeric_cols) < 3:
            st.warning("Need at least 3 numeric columns for a 3D plot.")
            return
        x_col = st.selectbox("X Axis", numeric_cols, key="3d_x")
        y_col = st.selectbox("Y Axis", numeric_cols, key="3d_y")
        z_col = st.selectbox("Z Axis", numeric_cols, key="3d_z")
        fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col, title="3D Scatter Plot")
        st.plotly_chart(fig, use_container_width=True)
    elif option == "Cluster Plot":
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        if numeric_df.shape[1] < 2:
            st.warning("Need at least 2 numeric features for clustering.")
            return
        k = st.slider("Number of clusters", min_value=2, max_value=10, value=3, key="clust_k")
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(numeric_df)
        numeric_df['Cluster'] = clusters.astype(str)
        fig = px.scatter_matrix(numeric_df, color="Cluster", title="Cluster Plot")
        st.plotly_chart(fig, use_container_width=True)

def plot_time_series(df):
    st.subheader("Time Series Visualization")
    date_col = st.selectbox("Select datetime column", df.columns, key="ts_date")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if date_col not in df.columns:
        st.warning("Invalid datetime column.")
        return
    try:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    except Exception as e:
        st.error(f"Error converting to datetime: {e}")
        return
    num_col = st.selectbox("Select numeric column", numeric_cols, key="ts_num")
    fig = px.line(df, x=date_col, y=num_col, title="Time Series Plot")
    st.plotly_chart(fig, use_container_width=True)

def plot_categorical(df):
    st.subheader("Categorical Data Visualization")
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if not cat_cols:
        st.warning("No categorical columns found.")
        return
    col = st.selectbox("Select categorical column", cat_cols, key="cat_col")
    option = st.selectbox("Select plot type", ["Count Plot", "Pie Chart"], key="cat_plot")
    if option == "Count Plot":
        counts = df[col].value_counts().reset_index()
        counts.columns = [col, "count"]
        fig = px.bar(counts, x=col, y="count", title=f"Count Plot of {col}")
    else:
        counts = df[col].value_counts().reset_index()
        counts.columns = [col, "count"]
        fig = px.pie(counts, names=col, values="count", title=f"Pie Chart of {col}")
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# Machine Learning Functions with Test Case and Download Options
# =============================================================================
def train_regression_model(df):
    st.subheader("Regression Model Training")
    target = st.selectbox("Select target column for regression", df.columns, key="reg_target")
    try:
        X = df.drop(columns=[target])
        y = df[target]
    except Exception as e:
        st.error(f"Error splitting features and target: {e}")
        return None, None, None
    test_size = st.slider("Select test size (%)", min_value=10, max_value=50, value=20, step=5, key="reg_test")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)
    
    model_choice = st.selectbox("Select Regression Model", 
                                ["Linear Regression", "Polynomial Regression (Degree 2)",
                                 "Ridge Regression", "Decision Tree Regressor", "Random Forest Regressor"],
                                key="reg_model")
    if model_choice == "Linear Regression":
        model = LinearRegression()
    elif model_choice == "Polynomial Regression (Degree 2)":
        poly = PolynomialFeatures(degree=2)
        X_train = poly.fit_transform(X_train)
        X_test = poly.transform(X_test)
        model = LinearRegression()
    elif model_choice == "Ridge Regression":
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1.0)
    elif model_choice == "Decision Tree Regressor":
        model = DecisionTreeRegressor(random_state=42)
    elif model_choice == "Random Forest Regressor":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    st.write("**Model Performance**")
    st.write(f"Mean Squared Error: {mse:.3f}")
    st.write(f"R² Score: {r2:.3f}")
    
    # Plot actual vs predicted
    results_df = pd.DataFrame({"Actual": y_test, "Predicted": predictions})
    fig = px.scatter(results_df, x="Actual", y="Predicted", trendline="ols", title="Actual vs Predicted")
    st.plotly_chart(fig, use_container_width=True)
    
    # --- Test Model Section ---
    st.markdown("### Test the Model with a Custom Input")
    st.write("Enter values for each feature (as used in the preprocessed data):")
    test_input = {}
    for col in X.columns:
        test_input[col] = st.number_input(f"Value for {col}", value=float(X[col].mean()), key=f"test_{col}")
    if st.button("Predict Test Case", key="reg_predict"):
        test_df = pd.DataFrame([test_input])
        # If polynomial transformation was applied, transform the input as well.
        if model_choice == "Polynomial Regression (Degree 2)":
            test_df = poly.transform(test_df)
        pred = model.predict(test_df)
        st.write("Prediction:", pred[0])
    
    # --- Download Model & Analysis Report ---
    model_pickle = pickle.dumps(model)
    st.download_button(label="Download Model as Pickle",
                       data=model_pickle,
                       file_name="regression_model.pkl",
                       mime="application/octet-stream")
    
    # Generate analysis report
    transformation = st.session_state.get("transform", "None")
    outlier_method = st.session_state.get("outliers", "None")
    reduction = st.session_state.get("reduce", "None")
    report = f"""Project Analysis Report
---------------------------
Dataset Shape: {df.shape}
Preprocessing Options:
    Transformation: {transformation}
    Outlier Handling: {outlier_method}
    Data Reduction: {reduction}

Regression Model Details:
    Model Chosen: {model_choice}
    Test Size: {test_size}%
    Performance Metrics:
        Mean Squared Error: {mse:.3f}
        R² Score: {r2:.3f}
"""
    st.download_button(label="Download Analysis Report",
                       data=report,
                       file_name="analysis_report.txt",
                       mime="text/plain")
    return model, X.columns, report

def train_classification_model(df):
    st.subheader("Classification Model Training")
    target = st.selectbox("Select target column for classification", df.columns, key="clf_target")
    try:
        X = df.drop(columns=[target])
        y = df[target]
    except Exception as e:
        st.error(f"Error splitting features and target: {e}")
        return None, None, None
    test_size = st.slider("Select test size (%)", min_value=10, max_value=50, value=20, step=5, key="clf_test")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)
    
    model_choice = st.selectbox("Select Classification Model", 
                                ["Logistic Regression", "Decision Tree Classifier", "Random Forest Classifier"],
                                key="clf_model")
    if model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_choice == "Decision Tree Classifier":
        model = DecisionTreeClassifier(random_state=42)
    elif model_choice == "Random Forest Classifier":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    st.write("**Model Performance**")
    st.write(f"Accuracy: {acc:.3f}")
    st.write("Confusion Matrix:")
    st.write(cm)
    st.text(classification_report(y_test, predictions))
    
    # Plot confusion matrix using Plotly
    cm_df = pd.DataFrame(cm)
    fig = px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues",
                    labels=dict(x="Predicted", y="Actual"),
                    x=[f"Class {i}" for i in range(cm.shape[0])],
                    y=[f"Class {i}" for i in range(cm.shape[0])],
                    title="Confusion Matrix")
    st.plotly_chart(fig, use_container_width=True)
    
    # --- Test Model Section ---
    st.markdown("### Test the Model with a Custom Input")
    st.write("Enter values for each feature (as used in the preprocessed data):")
    test_input = {}
    for col in X.columns:
        test_input[col] = st.number_input(f"Value for {col}", value=float(X[col].mean()), key=f"test_{col}")
    if st.button("Predict Test Case", key="clf_predict"):
        test_df = pd.DataFrame([test_input])
        pred = model.predict(test_df)
        st.write("Prediction:", pred[0])
    
    # --- Download Model & Analysis Report ---
    model_pickle = pickle.dumps(model)
    st.download_button(label="Download Model as Pickle",
                       data=model_pickle,
                       file_name="classification_model.pkl",
                       mime="application/octet-stream")
    
    transformation = st.session_state.get("transform", "None")
    outlier_method = st.session_state.get("outliers", "None")
    reduction = st.session_state.get("reduce", "None")
    report = f"""Project Analysis Report
---------------------------
Dataset Shape: {df.shape}
Preprocessing Options:
    Transformation: {transformation}
    Outlier Handling: {outlier_method}
    Data Reduction: {reduction}

Classification Model Details:
    Model Chosen: {model_choice}
    Test Size: {test_size}%
    Performance Metrics:
        Accuracy: {acc:.3f}
"""
    st.download_button(label="Download Analysis Report",
                       data=report,
                       file_name="analysis_report.txt",
                       mime="text/plain")
    return model, X.columns, report


# =============================================================================
# Main Application Layout
# =============================================================================
def main():
    st.title("Data Analysis & Machine Learning Web App")
    st.markdown("""
    **Workflow:**
    1. Upload and preprocess your dataset once.
    2. Use the preprocessed data for visualization and model training.
    3. Test the model with custom input, download the trained model, and obtain an analysis report.
    """)
    
    # Sidebar: Data Upload and Preprocessing
    st.sidebar.header("1. Upload & Preprocess Data")
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"], key="upload")
    encoding = st.sidebar.selectbox("Select file encoding", ["utf-8", "ISO-8859-1", "ascii", "utf-16", "cp1252"], key="encoding")
    
    if uploaded_file is not None:
        # Load data only once and store in session_state
        if "raw_data" not in st.session_state:
            raw_df = load_data(uploaded_file, encoding)
            if raw_df is not None:
                st.session_state["raw_data"] = raw_df
        else:
            raw_df = st.session_state["raw_data"]
        
        st.sidebar.write("### Raw Data Preview")
        st.sidebar.dataframe(raw_df.head())
        
        # Preprocessing options (including outlier management)
        st.sidebar.header("Preprocessing Options")
        preprocessed_df = preprocess_data(raw_df.copy())
        st.session_state["preprocessed_data"] = preprocessed_df
        
        st.sidebar.write("### Preprocessed Data Preview")
        st.sidebar.dataframe(preprocessed_df.head())
    else:
        st.info("Please upload a dataset using the sidebar to get started.")
        st.stop()  # Stops further execution until file is uploaded.
    
    # Navigation among app sections
    menu = st.sidebar.radio("2. Choose Section", ["Data Visualization", "Machine Learning"])
    
    if menu == "Data Visualization":
        st.header("Data Visualization")
        vis_option = st.selectbox("Select Visualization Type", 
                                  ["Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis", 
                                   "Time Series Visualization", "Categorical Data Visualization"],
                                  key="vis_option")
        df = st.session_state["preprocessed_data"]
        if vis_option == "Univariate Analysis":
            plot_univariate(df)
        elif vis_option == "Bivariate Analysis":
            plot_bivariate(df)
        elif vis_option == "Multivariate Analysis":
            plot_multivariate(df)
        elif vis_option == "Time Series Visualization":
            plot_time_series(df)
        elif vis_option == "Categorical Data Visualization":
            plot_categorical(df)
    
    elif menu == "Machine Learning":
        st.header("Machine Learning Model Training")
        problem_type = st.selectbox("Select Problem Type", ["Regression", "Classification"], key="problem_type")
        df = st.session_state["preprocessed_data"]
        if problem_type == "Regression":
            train_regression_model(df)
        else:
            train_classification_model(df)

if __name__ == '__main__':
    main()
