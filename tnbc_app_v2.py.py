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
import gdown
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
SAMPLE_FILE_ID = "1lWl8QWQuzo2PxK0Mft6j7aETVDyDASaU"  # This is feature_importance.csv

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="TNBC Drug Discovery",
    page_icon="💊",
    layout="wide"
)

st.title("💊 TNBC Drug Discovery - Model Inference")
st.markdown("""
This app uses a pre-trained machine learning model to predict activity scores for
Triple-Negative Breast Cancer (TNBC) drug candidates. Upload your compound data
to get ranked predictions.
""")

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def load_model_with_fallback(model_path):
    """Try multiple methods to load the model"""
    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        return data, "pickle"
    except Exception as e:
        st.warning(f"Standard pickle loading failed: {str(e)}")

    try:
        data = joblib.load(model_path)
        return data, "joblib"
    except Exception as e:
        st.warning(f"Joblib loading failed: {str(e)}")

    return None, None

# ==========================================
# NEW: Smart file reader for multiple formats
# ==========================================
def read_uploaded_file(uploaded_file):
    """
    Reads Excel, CSV, TSV, TXT, and other text-based tabular files.
    Auto-detects delimiters for text files.
    """
    file_name = uploaded_file.name.lower()
    
    # --- Excel files ---
    if file_name.endswith(('.xlsx', '.xls')):
        return pd.read_excel(uploaded_file)
    
    # --- TSV files (Tab-separated) ---
    if file_name.endswith('.tsv'):
        return pd.read_csv(uploaded_file, sep='\t')
    
    # --- CSV and TXT files (auto-detect delimiter) ---
    if file_name.endswith(('.csv', '.txt')):
        # Try auto-detecting delimiter (comma, semicolon, tab, space, pipe)
        try:
            # First, read a small sample to detect the delimiter
            sample = uploaded_file.getvalue().decode('utf-8')[:1024]
            
            # Count occurrences of potential delimiters in the first line
            lines = sample.split('\n')
            if lines:
                first_line = lines[0]
                delimiters = {
                    ',': first_line.count(','),
                    ';': first_line.count(';'),
                    '\t': first_line.count('\t'),
                    '|': first_line.count('|'),
                    ' ': first_line.count(' ')  # space (less common but possible)
                }
                # Choose the delimiter with the most occurrences (ignore spaces if too many)
                best_delim = max(delimiters, key=delimiters.get)
                # If space is chosen but there are very few, default to comma
                if best_delim == ' ' and delimiters[' '] < 3:
                    best_delim = ','
                
                # Reset file pointer to beginning
                uploaded_file.seek(0)
                
                # Read with detected delimiter
                return pd.read_csv(uploaded_file, sep=best_delim, engine='python')
                
        except Exception as e:
            # Fallback: try comma, then tab, then semicolon
            uploaded_file.seek(0)
            try:
                return pd.read_csv(uploaded_file)
            except:
                uploaded_file.seek(0)
                try:
                    return pd.read_csv(uploaded_file, sep='\t')
                except:
                    uploaded_file.seek(0)
                    return pd.read_csv(uploaded_file, sep=';')
    
    # Fallback
    return None

# ==========================================
# SIDEBAR (Model Loading - UNCHANGED)
# ==========================================
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("---")

    st.subheader("Model Loading")
    
    uploaded_model = st.file_uploader(
        "Upload model file",
        type=['pkl', 'joblib'],
        help="Upload your trained model file"
    )

    if uploaded_model is not None:
        temp_path = "temp_model." + uploaded_model.name.split('.')[-1]
        with open(temp_path, 'wb') as f:
            f.write(uploaded_model.getbuffer())
        model_path = temp_path
        st.success(f"✅ Uploaded: {uploaded_model.name}")
        
        try:
            model_data, method_used = load_model_with_fallback(model_path)
            
            if model_data is not None:
                st.success(f"✅ Model loaded successfully!")
                
                if isinstance(model_data, dict):
                    if 'model' in model_data:
                        st.session_state['model'] = model_data['model']
                        if 'scaler' in model_data:
                            st.session_state['scaler'] = model_data['scaler']
                        else:
                            st.session_state['scaler'] = StandardScaler()
                    else:
                        for key, value in model_data.items():
                            if hasattr(value, 'predict'):
                                st.session_state['model'] = value
                                break
                        st.session_state['scaler'] = StandardScaler()
                else:
                    st.session_state['model'] = model_data
                    st.session_state['scaler'] = StandardScaler()
                
                model = st.session_state['model']
                if hasattr(model, 'n_features_in_'):
                    st.session_state['expected_features'] = model.n_features_in_
                    st.info(f"Model expects {model.n_features_in_} features")
                
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")

    model_loaded = 'model' in st.session_state
    
    if model_loaded:
        st.success("✅ Model ready for inference")
        if 'expected_features' in st.session_state:
            st.info(f"Expected features: {st.session_state['expected_features']}")
    else:
        st.warning("⚠️ Please upload a model file")

    st.markdown("---")

    st.subheader("Display Options")
    num_results = st.slider("Number of results to show", min_value=5, max_value=50, value=20)
    show_stats = st.checkbox("Show statistics", value=True)
    show_plots = st.checkbox("Show plots", value=True)

# ==========================================
# MAIN TABS
# ==========================================
tab1, tab2, tab3 = st.tabs(["📤 Upload & Predict", "📊 Results", "ℹ️ About"])

# ==========================================
# TAB 1: UPLOAD & PREDICT (UPDATED)
# ==========================================
with tab1:
    st.header("Load Your Data")

    if not model_loaded:
        st.warning("⚠️ Please upload a model file in the sidebar first.")
    else:
        # Two-column layout for Sample vs Upload
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📂 Option 1: Load Sample")
            if st.button("📥 Load Sample Dataset from Drive", use_container_width=True):
                if SAMPLE_FILE_ID == "YOUR_FILE_ID_HERE":
                    st.warning("⚠️ Please set the SAMPLE_FILE_ID at the top of the script.")
                else:
                    with st.spinner("Downloading sample dataset from Google Drive..."):
                        try:
                            url = f"https://drive.google.com/uc?id={SAMPLE_FILE_ID}"
                            df = gdown.load_as_dataframe(url)
                            st.session_state['raw_data'] = df
                            st.success(f"✅ Loaded {len(df)} rows!")
                            st.dataframe(df.head())
                        except Exception as e:
                            st.error(f"Failed to load sample: {str(e)}")

        with col2:
            st.subheader("📤 Option 2: Upload File")
            
            # --- UPDATED: Support for more file types ---
            uploaded_file = st.file_uploader(
                "Choose your file (Excel, CSV, TSV, TXT)",
                type=['xlsx', 'xls', 'csv', 'tsv', 'txt'],  # <-- UPDATED
                help="Upload Excel, CSV, TSV, or any text-based tabular file.",
                label_visibility="collapsed"
            )

            if uploaded_file is not None:
                try:
                    # --- UPDATED: Use the smart reader ---
                    df = read_uploaded_file(uploaded_file)
                    
                    if df is not None:
                        st.session_state['raw_data'] = df
                        st.success(f"✅ Loaded {len(df)} rows!")
                        st.dataframe(df.head())
                    else:
                        st.error("Could not read the file. Please check the format.")
                        
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")

        st.divider()

        # --- SHARED FEATURE SELECTION & PREDICTION ---
        if 'raw_data' in st.session_state and st.session_state['raw_data'] is not None:
            df = st.session_state['raw_data']
            
            st.subheader("📊 Data Preview")
            st.dataframe(df.head(10))
            st.info(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")

            st.subheader("🔍 Feature Selection")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            expected_features = st.session_state.get('expected_features', len(numeric_cols))
            
            st.write(f"Available numeric columns: {len(numeric_cols)}")
            st.write(f"Model expects: {expected_features} features")
            
            # --- UPDATED: Better handling for non-molecule files ---
            if len(numeric_cols) == 0:
                st.warning("⚠️ This file contains no numeric columns. It may be a feature importance file or a text report. You can view it here, but it cannot be used for predictions.")
                # Show the full data for viewing
                st.dataframe(df)
            elif len(numeric_cols) >= expected_features:
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
                    
                    st.write("Selected features (in order):")
                    for i, col in enumerate(feature_cols):
                        st.write(f"{i+1}. {col}")
                    
                    if st.button("🚀 Run Predictions", type="primary", use_container_width=True):
                        with st.spinner("Running predictions..."):
                            try:
                                X = df[feature_cols].values
                                
                                if np.any(pd.isnull(X)):
                                    st.warning("⚠️ Missing values detected. Filling with column means.")
                                    imputer = SimpleImputer(strategy='mean')
                                    X = imputer.fit_transform(X)
                                
                                if 'scaler' in st.session_state:
                                    if hasattr(st.session_state['scaler'], 'fit'):
                                        X_scaled = st.session_state['scaler'].fit_transform(X)
                                    else:
                                        X_scaled = st.session_state['scaler'].transform(X)
                                else:
                                    scaler = StandardScaler()
                                    X_scaled = scaler.fit_transform(X)
                                
                                predictions = st.session_state['model'].predict(X_scaled)
                                
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
                st.warning(f"⚠️ Your file has {len(numeric_cols)} numeric columns, but the model expects {expected_features}. You can still view the data, but predictions won't work with this file.")
                st.info("💡 If this is a **feature importance** file (like the one you uploaded), it's meant for display, not prediction. Upload a molecule dataset with descriptor columns instead.")

# ==========================================
# TAB 2: RESULTS (UNCHANGED)
# ==========================================
with tab2:
    st.header("Prediction Results")

    if 'results_df' in st.session_state:
        results_df = st.session_state['results_df']
        
        st.subheader(f"🏆 Top {num_results} Ranked Compounds")
        display_df = results_df.head(num_results).copy()
        display_df['Predicted_Activity'] = display_df['Predicted_Activity'].map('{:.4f}'.format)
        st.dataframe(display_df, use_container_width=True, height=400)
        
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results",
            data=csv,
            file_name="tnbc_predictions.csv",
            mime="text/csv"
        )
        
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

# ==========================================
# TAB 3: ABOUT (UNCHANGED)
# ==========================================
with tab3:
    st.header("ℹ️ About This App")
    st.markdown("""
    ### TNBC Drug Discovery Pipeline
    
    **How to use:**
    1. Upload your trained model file (`.pkl` or `.joblib`) in the sidebar.
    2. Load your compound data via **Option 1** (sample from Drive) or **Option 2** (upload file).
    3. Select the columns to use for prediction.
    4. Click "Run Predictions".
    
    **Supported file formats:**
    - Excel (`.xlsx`, `.xls`)
    - CSV (`.csv`) – auto-detects commas, semicolons, tabs, or pipes
    - TSV (`.tsv`)
    - Text (`.txt`) – auto-detects delimiters
    
    **Note:** The model expects a specific number of features. 
    Select your columns in the same order they were used during training.
    """)

# Footer
st.divider()
st.caption("🚀 TNBC Drug Discovery Pipeline | Created with Streamlit | For Research Use Only")
