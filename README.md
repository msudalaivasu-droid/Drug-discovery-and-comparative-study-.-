# 💊 TNBC Drug Discovery & Comparative Study

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

## 🧪 About the Project
This is an interactive **Machine Learning Inference App** built for predicting activity scores of drug candidates against **Triple-Negative Breast Cancer (TNBC)**. 

Users can upload their own molecular descriptor datasets (Excel, CSV, TSV, or TXT) and a pre-trained `.pkl` model to get ranked predictions instantly.

## ✨ Key Features
- 📤 **Multi-format Uploader**: Supports `.xlsx`, `.csv`, `.tsv`, and `.txt` (auto-detects delimiters).
- 🤖 **Model Inference**: Upload your trained `sklearn` or `joblib` model.
- 📊 **Interactive Visualization**: View histograms, top-10 bar charts, and detailed statistics.
- ⬇️ **Download Results**: Export predictions as a CSV file.

## 🛠️ Tech Stack
- **Frontend/UI**: Streamlit
- **Data Handling**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **ML Backend**: Scikit-learn, Joblib

## 🚀 Live Demo
[Click here to view the live app](https://your-app-url-here.streamlit.app) *(Replace this with your actual Streamlit URL after deployment!)*

## 📂 How to Run Locally
1. Clone the repo:
   ```bash
   git clone https://github.com/msudalaivasu-droid/Drug-discovery-and-comparative-study-.-.git






   
Install dependancies
pip install -r requirements.txt





Run the app:


streamlit run app.py
Disclaimer: For research and educational purposes only. Not for clinical use.




### 🚨 Final Check: The "Sample Data" Button Issue
Look closely at your `app.py` in the GitHub preview. If you see `SAMPLE_FILE_ID = "1lWl8QWQuzo2PxK0Mft6j7aETVDyDASaU"`, remember:

> **This is the ID for `feature_importance.csv` (the 2-column file).** 
> If you deploy right now, users will click "Load Sample" and get an error when they try to predict.

**Action Step**: 
1. Upload your **real** dataset (the big one with columns like `MW`, `TPSA`, `LogP`) to Google Drive.
2. Copy its File ID.
3. Go to GitHub, click on `app.py`, click the pencil icon (✏️) to edit it directly online.
4. Replace `1lWl8QWQuzo2PxK0Mft6j7aETVDyDASaU` with your new ID.
5. Scroll down and click **"Commit changes"**.

---

### 🎉 Success Check
If you see the green "Code" button, the files are listed, and the commit message is visible—**you have successfully made it to the final look!** 

You are now ready to click **"Deploy"** on Streamlit Cloud. Go to [share.streamlit.io](https://share.streamlit.io), log in, point it to this repo, and watch the magic happen!

Let me know when you've swapped out that Sample File ID or if you need help finding the right one! 🚀
