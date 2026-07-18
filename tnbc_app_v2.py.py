# ==========================================
# TNBC Drug Discovery - Streamlit Inference App
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="TNBC Drug Discovery",
    page_icon="💊",
    layout="wide"
)

# Title and description
st.title("💊 TNBC Drug Discovery - Model Inference")
st.markdown("""
This app uses a pre-trained machine learning model to predict activity scores for
Triple-Negative Breast Cancer (TNBC) drug candidates. Upload your compound data
to get ranked predictions.
""")

# Function to try loading model with different methods
def load_model_with_fallback(model_path):
    """Try multiple methods to load the model"""
    # Method 1: Try standard pickle
    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        return data, "pickle"
    except Exception as e:
        st.warning(f"Standard pickle loading failed: {str(e)}")

    # Method 2: Try joblib
    try:
        data = joblib.load(model_path)
        return data, "joblib"
    except Exception as e:
        st.warning(f"Joblib loading failed: {str(e)}")

    return None, None

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("---")

    # Model loading section
    st.subheader("Model Loading")
    
    # Option to upload model file
    uploaded_model = st.file_uploader(
        "Upload model file",
        type=['pkl', 'joblib'],
        help="Upload your trained model file"
    )

    if uploaded_model is not None:
        # Save uploaded file temporarily
        temp_path = "temp_model." + uploaded_model.name.split('.')[-1]
        with open(temp_path, 'wb') as f:
            f.write(uploaded_model.getbuffer())
        model_path = temp_path
        st.success(f"✅ Uploaded: {uploaded_model.name}")
        
        # Try to load model
        try:
            model_data, method_used = load_model_with_fallback(model_path)
            
            if model_data is not None:
                st.success(f"✅ Model loaded successfully!")
                
                # Handle different model formats
                if isinstance(model_data, dict):
                    if 'model' in model_data:
                        st.session_state['model'] = model_data['model']
                        if 'scaler' in model_data:
                            st.session_state['scaler'] = model_data['scaler']
                        else:
                            st.session_state['scaler'] = StandardScaler()
                    else:
                        # Try to find model in dict
                        for key, value in model_data.items():
                            if hasattr(value, 'predict'):
                                st.session_state['model'] = value
                                break
                        st.session_state['scaler'] = StandardScaler()
                else:
                    st.session_state['model'] = model_data
                    st.session_state['scaler'] = StandardScaler()
                
                # Get model info
                model = st.session_state['model']
                if hasattr(model, 'n_features_in_'):
                    st.session_state['expected_features'] = model.n_features_in_
                    st.info(f"Model expects {model.n_features_in_} features")
                
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")

    # Check if model is loaded
    model_loaded = 'model' in st.session_state
    
    if model_loaded:
        st.success("✅ Model ready for inference")
        if 'expected_features' in st.session_state:
            st.info(f"Expected features: {st.session_state['expected_features']}")
    else:
        st.warning("⚠️ Please upload a model file")

    st.markdown("---")

    # Display options
    st.subheader("Display Options")
    num_results = st.slider("Number of results to show", min_value=5, max_value=50, value=20)
    show_stats = st.checkbox("Show statistics", value=True)
    show_plots = st.checkbox("Show plots", value=True)

# Main content area
tab1, tab2, tab3 = st.tabs(["📤 Upload & Predict", "📊 Results", "ℹ️ About"])

with tab1:
    st.header("Upload Your Data")

    if not model_loaded:
        st.warning("⚠️ Please upload a model file in the sidebar first.")
    else:
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a data file containing drug compounds",
            type=['xlsx', 'xls', 'csv'],
            help="Upload an Excel or CSV file with molecular descriptors"
        )

        if uploaded_file is not None:
            try:
                # Load data
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                st.session_state['raw_data'] = df
                
                # Data preview
                st.subheader("Data Preview")
                st.dataframe(df.head(10))
                st.info(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")

                # Feature selection
                st.subheader("🔍 Feature Selection")
                
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                expected_features = st.session_state.get('expected_features', len(numeric_cols))
                
                st.write(f"Available numeric columns: {len(numeric_cols)}")
                st.write(f"Model expects: {expected_features} features")
                
                if len(numeric_cols) >= expected_features:
                    # Simple column selection
                    st.write("Select the columns to use for prediction:")
                    
                    # If we have exactly the right number, suggest them
                    if len(numeric_cols) == expected_features:
                        default_cols = numeric_cols
                    else:
                        default_cols = numeric_cols[:expected_features]
                    
                    feature_cols = st.multiselect(
                        f"Choose exactly {expected_features} columns",
                        options=numeric_cols,
                        default=default_cols
                    )
                    
                    if len(feature_cols) == expected_features:
                        st.success(f"✅ Selected {len(feature_cols)} features")
                        
                        # Show selected order
                        st.write("Selected features (in order):")
                        for i, col in enumerate(feature_cols):
                            st.write(f"{i+1}. {col}")
                        
                        # Prediction button
                        if st.button("🚀 Run Predictions", type="primary", use_container_width=True):
                            with st.spinner("Running predictions..."):
                                try:
                                    # Prepare features
                                    X = df[feature_cols].values
                                    
                                    # Handle missing values
                                    if np.any(pd.isnull(X)):
                                        st.warning("⚠️ Missing values detected. Filling with column means.")
                                        imputer = SimpleImputer(strategy='mean')
                                        X = imputer.fit_transform(X)
                                    
                                    # Scale features
                                    if 'scaler' in st.session_state:
                                        if hasattr(st.session_state['scaler'], 'fit'):
                                            X_scaled = st.session_state['scaler'].fit_transform(X)
                                        else:
                                            X_scaled = st.session_state['scaler'].transform(X)
                                    else:
                                        scaler = StandardScaler()
                                        X_scaled = scaler.fit_transform(X)
                                    
                                    # Make predictions
                                    predictions = st.session_state['model'].predict(X_scaled)
                                    
                                    # Create results
                                    results_df = df.copy()
                                    results_df['Predicted_Activity'] = predictions
                                    results_df['Rank'] = results_df['Predicted_Activity'].rank(
                                        method="dense", ascending=False
                                    ).astype(int)
                                    results_df = results_df.sort_values('Rank')
                                    
                                    st.session_state['results_df'] = results_df
                                    
                                    st.success("✅ Predictions completed!")
                                    st.balloons()
                                    st.markdown("👉 Go to the **Results** tab")
                                    
                                except Exception as e:
                                    st.error(f"Prediction error: {str(e)}")
                    else:
                        st.warning(f"Please select exactly {expected_features} columns")
                else:
                    st.error(f"Your data has only {len(numeric_cols)} numeric columns, but model expects {expected_features}")
                    
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")

with tab2:
    st.header("Prediction Results")

    if 'results_df' in st.session_state:
        results_df = st.session_state['results_df']
        
        # Top results
        st.subheader(f"🏆 Top {num_results} Ranked Compounds")
        display_df = results_df.head(num_results).copy()
        display_df['Predicted_Activity'] = display_df['Predicted_Activity'].map('{:.4f}'.format)
        st.dataframe(display_df, use_container_width=True, height=400)
        
        # Download
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results",
            data=csv,
            file_name="tnbc_predictions.csv",
            mime="text/csv"
        )
        
        # Statistics
        if show_stats:
            st.subheader("📊 Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean", f"{results_df['Predicted_Activity'].mean():.4f}")
            with col2:
                st.metric("Median", f"{results_df['Predicted_Activity'].median():.4f}")
            with col3:
                st.metric("Max", f"{results_df['Predicted_Activity'].max():.4f}")
            with col4:
                st.metric("Min", f"{results_df['Predicted_Activity'].min():.4f}")
        
        # Plots
        if show_plots:
            st.subheader("📈 Visualization")
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots()
                sns.histplot(results_df['Predicted_Activity'], bins=30, kde=True, ax=ax)
                ax.set_title("Distribution of Predictions")
                st.pyplot(fig)
                plt.close()
            
            with col2:
                fig, ax = plt.subplots()
                top10 = results_df.head(10)
                
                # Try to find ID column
                id_col = None
                for col in ['Compound_Name', 'Drug_Name', 'Compound_ID', 'ID', 'Name']:
                    if col in results_df.columns:
                        id_col = col
                        break
                
                if id_col:
                    labels = top10[id_col].values
                else:
                    labels = [f"Rank {i+1}" for i in range(len(top10))]
                
                y_pos = range(len(top10))
                bars = ax.barh(y_pos, top10['Predicted_Activity'].values)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(labels)
                ax.set_xlabel("Predicted Activity")
                ax.set_title("Top 10 Compounds")
                ax.invert_yaxis()
                
                for i, (bar, val) in enumerate(zip(bars, top10['Predicted_Activity'].values)):
                    ax.text(val, i, f' {val:.4f}', va='center')
                
                st.pyplot(fig)
                plt.close()
    else:
        st.info("No predictions yet. Upload data and run predictions in the Upload tab.")

with tab3:
    st.header("ℹ️ About This App")
    st.markdown("""
    ### TNBC Drug Discovery Pipeline
    
    **How to use:**
    1. Upload your trained model file (`.pkl` or `.joblib`)
    2. Upload your compound data (Excel or CSV)
    3. Select the columns to use for prediction
    4. Click "Run Predictions"
    
    **Note:** The model expects a specific number of features. 
    Select your columns in the same order they were used during training.
    """)

# Footer
st.divider()
st.caption("🚀 TNBC Drug Discovery Pipeline | Created with Streamlit | For Research Use Only")