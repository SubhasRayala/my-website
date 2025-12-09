import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="ML Model Trainer", layout="wide")

# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state.data = None
if 'input_cols' not in st.session_state:
    st.session_state.input_cols = []
if 'target_col' not in st.session_state:
    st.session_state.target_col = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'problem_type' not in st.session_state:
    st.session_state.problem_type = None
if 'categorical_columns' not in st.session_state:
    st.session_state.categorical_columns = []
if 'encoded_column_names' not in st.session_state:
    st.session_state.encoded_column_names = []
if 'categorical_mappings' not in st.session_state:
    st.session_state.categorical_mappings = {}

# Title
st.title("🤖 Machine Learning Model Trainer & Predictor")
st.markdown("---")

# Step 1: Upload CSV File
st.header("📁 Step 1: Upload Your Dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

if uploaded_file is not None:
    # Read the CSV file
    st.session_state.data = pd.read_csv(uploaded_file)
    
    st.success(f"✅ Dataset uploaded successfully! Shape: {st.session_state.data.shape}")
    
    # Display columns in rows
    st.subheader("📊 Columns in Dataset:")
    for idx, col in enumerate(st.session_state.data.columns, 1):
        data_type = "Categorical (Text)" if st.session_state.data[col].dtype == 'object' else f"Numeric ({st.session_state.data[col].dtype})"
        st.write(f"{idx}. **{col}** - Data Type: {data_type}")
    
    st.markdown("---")
    
    # Step 2: Select Input and Target Columns
    st.header("🎯 Step 2: Select Input & Target Columns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Columns (Features)")
        st.caption(f"Select minimum 1 and maximum {len(st.session_state.data.columns)-1} columns")
        
        # Search bar for input columns
        search_input = st.text_input("🔍 Search input columns", key="search_input")
        
        # Filter columns based on search
        if search_input:
            filtered_cols = [col for col in st.session_state.data.columns if search_input.lower() in col.lower()]
        else:
            filtered_cols = list(st.session_state.data.columns)
        
        # Checkboxes for input columns
        selected_inputs = []
        for col in filtered_cols:
            if st.checkbox(col, key=f"input_{col}"):
                selected_inputs.append(col)
        
        st.session_state.input_cols = selected_inputs
        
        # Display selected input columns vertically
        if st.session_state.input_cols:
            st.markdown("**Selected Input Columns:**")
            for col in st.session_state.input_cols:
                st.write(f"✓ {col}")
    
    with col2:
        st.subheader("Target Column (Output)")
        st.caption("Select exactly 1 column")
        
        # Search bar for target column
        search_target = st.text_input("🔍 Search target column", key="search_target")
        
        # Filter columns based on search (exclude already selected input columns)
        available_target_cols = [col for col in st.session_state.data.columns if col not in st.session_state.input_cols]
        
        if search_target:
            filtered_target_cols = [col for col in available_target_cols if search_target.lower() in col.lower()]
        else:
            filtered_target_cols = available_target_cols
        
        # Radio buttons for target column (only one selection)
        if filtered_target_cols:
            st.session_state.target_col = st.radio("Select target column:", filtered_target_cols, key="target_radio")
        
        if st.session_state.target_col:
            st.markdown(f"**Selected Target Column:**")
            st.write(f"✓ {st.session_state.target_col}")
    
    st.markdown("---")
    
    # Step 3: Detect Problem Type and Select Model
    if st.session_state.input_cols and st.session_state.target_col:
        
        # Validate selection
        max_input_cols = len(st.session_state.data.columns) - 1
        if len(st.session_state.input_cols) < 1 or len(st.session_state.input_cols) > max_input_cols:
            st.error(f"⚠️ Please select between 1 and {max_input_cols} input columns!")
        else:
            st.header("🔧 Step 3: Select Machine Learning Model")
            
            # Detect problem type based on target column
            target_data = st.session_state.data[st.session_state.target_col]
            unique_values = target_data.nunique()
            
            # Determine if categorical or continuous
            if target_data.dtype == 'object' or unique_values < 20:
                st.session_state.problem_type = "Classification"
                suggested_models = {
                    "Logistic Regression": LogisticRegression(max_iter=1000),
                    "Random Forest Classifier": RandomForestClassifier(random_state=42),
                    "Decision Tree Classifier": DecisionTreeClassifier(random_state=42)
                }
                st.info(f"🎯 **Detected Problem Type:** Classification (Target has {unique_values} unique values)")
            else:
                st.session_state.problem_type = "Regression"
                suggested_models = {
                    "Linear Regression": LinearRegression(),
                    "Random Forest Regressor": RandomForestRegressor(random_state=42),
                    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42)
                }
                st.info(f"📈 **Detected Problem Type:** Regression (Target is continuous)")
            
            # Model selection dropdown
            model_name = st.selectbox("Select a model:", list(suggested_models.keys()))
            
            st.markdown("---")
            
            # Step 4: Train Model
            st.header("🚀 Step 4: Train Your Model")
            
            if st.button("🎓 Train Model", type="primary"):
                with st.spinner("Training model... Please wait ⏳"):
                    try:
                        # Prepare data
                        X = st.session_state.data[st.session_state.input_cols].copy()
                        y = st.session_state.data[st.session_state.target_col].copy()
                        
                        # Identify categorical columns in input features
                        st.session_state.categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
                        
                        # Store original categorical values for prediction dropdown
                        st.session_state.categorical_mappings = {}
                        for col in st.session_state.categorical_columns:
                            st.session_state.categorical_mappings[col] = X[col].unique().tolist()
                        
                        # Apply One-Hot Encoding to categorical columns
                        if st.session_state.categorical_columns:
                            st.info(f"🔄 Applying One-Hot Encoding to categorical columns: {', '.join(st.session_state.categorical_columns)}")
                            X = pd.get_dummies(X, columns=st.session_state.categorical_columns, dtype=int)
                        
                        # Store encoded column names for prediction
                        st.session_state.encoded_column_names = X.columns.tolist()
                        
                        # Handle missing values in numeric columns
                        X = X.fillna(X.mean(numeric_only=True))
                        
                        # For classification, encode target if it's categorical
                        if st.session_state.problem_type == "Classification" and y.dtype == 'object':
                            y = pd.Categorical(y).codes
                        
                        # Split data
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        
                        # Train model
                        st.session_state.model = suggested_models[model_name]
                        st.session_state.model.fit(X_train, y_train)
                        
                        # Evaluate model
                        y_pred = st.session_state.model.predict(X_test)
                        
                        if st.session_state.problem_type == "Classification":
                            score = accuracy_score(y_test, y_pred)
                            metric_name = "Accuracy"
                        else:
                            score = r2_score(y_test, y_pred)
                            metric_name = "R² Score"
                        
                        st.session_state.model_trained = True
                        st.success(f"✅ Model trained successfully!")
                        st.metric(label=f"Model {metric_name}", value=f"{score:.4f}")
                        
                        # Display encoding information
                        if st.session_state.categorical_columns:
                            with st.expander("ℹ️ One-Hot Encoding Details"):
                                st.write(f"**Original Features:** {len(st.session_state.input_cols)}")
                                st.write(f"**Encoded Features:** {len(st.session_state.encoded_column_names)}")
                                st.write(f"**Encoded Column Names:**")
                                st.write(st.session_state.encoded_column_names)
                        
                    except Exception as e:
                        st.error(f"❌ Error during training: {str(e)}")
            
            # Step 5: Make Predictions
            if st.session_state.model_trained:
                st.markdown("---")
                st.header("🔮 Step 5: Make Predictions")
                
                # Create input fields for prediction
                prediction_inputs = {}
                
                st.subheader("Enter values for prediction:")
                
                # Create columns for better layout
                num_cols = 2
                cols = st.columns(num_cols)
                
                for idx, col_name in enumerate(st.session_state.input_cols):
                    with cols[idx % num_cols]:
                        # Check if column is categorical
                        if col_name in st.session_state.categorical_columns:
                            # Show dropdown for categorical columns
                            prediction_inputs[col_name] = st.selectbox(
                                f"**{col_name}**",
                                options=st.session_state.categorical_mappings[col_name],
                                key=f"pred_{col_name}"
                            )
                        else:
                            # Check data type for numeric columns
                            col_dtype = st.session_state.data[col_name].dtype
                            
                            if col_dtype in ['int64', 'int32']:
                                prediction_inputs[col_name] = st.number_input(
                                    f"**{col_name}**", 
                                    value=int(st.session_state.data[col_name].mean()),
                                    step=1,
                                    key=f"pred_{col_name}"
                                )
                            elif col_dtype in ['float64', 'float32']:
                                prediction_inputs[col_name] = st.number_input(
                                    f"**{col_name}**", 
                                    value=float(st.session_state.data[col_name].mean()),
                                    step=0.1,
                                    key=f"pred_{col_name}"
                                )
                
                st.markdown("---")
                
                if st.button("🎯 Predict", type="primary"):
                    try:
                        # Prepare input for prediction
                        input_df = pd.DataFrame([prediction_inputs])
                        
                        # Apply One-Hot Encoding to categorical columns (same as training)
                        if st.session_state.categorical_columns:
                            input_df = pd.get_dummies(input_df, columns=st.session_state.categorical_columns, dtype=int)
                            
                            # Ensure all columns from training are present
                            for col in st.session_state.encoded_column_names:
                                if col not in input_df.columns:
                                    input_df[col] = 0
                            
                            # Reorder columns to match training data
                            input_df = input_df[st.session_state.encoded_column_names]
                        
                        # Make prediction
                        prediction = st.session_state.model.predict(input_df)[0]
                        
                        # Display result in tabular format
                        st.subheader("📊 Prediction Results")
                        
                        # Create results dataframe
                        results_data = []
                        for col_name, value in prediction_inputs.items():
                            results_data.append({"Feature": col_name, "Input Value": value})
                        
                        results_df = pd.DataFrame(results_data)
                        
                        # Display inputs and prediction side by side
                        col_left, col_right = st.columns([2, 1])
                        
                        with col_left:
                            st.markdown("**Input Features:**")
                            st.dataframe(results_df, use_container_width=True, hide_index=True)
                        
                        with col_right:
                            st.markdown("**Predicted Value:**")
                            st.markdown(f"### {st.session_state.target_col}")
                            
                            if st.session_state.problem_type == "Regression":
                                st.success(f"# {prediction:.4f}")
                            else:
                                st.success(f"# {prediction}")
                        
                    except Exception as e:
                        st.error(f"❌ Error during prediction: {str(e)}")

else:
    st.info("👆 Please upload a CSV file to get started!")

# Footer
st.markdown("---")
st.markdown("Built by Subhas Rayala | Machine Learning Model Trainer")
