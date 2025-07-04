# Diabetic Retinopathy Detection

This project provides a web-based tool for detecting diabetic retinopathy from retinal images using deep learning. It features a Flask backend for model inference and a Streamlit frontend for user interaction.

## Features
- Upload retinal images for automated diabetic retinopathy detection
- Deep learning model (Keras/TensorFlow) for image classification
- Streamlit web interface for easy use
- Flask API backend for model serving
- Pre-trained model files included

## Project Structure
```
Diabetic_Retinopathy_Detection/
├── app.py                  # Flask backend (API server)
├── streamlit_app.py        # Streamlit frontend (main UI)
├── requirements.txt        # Python dependencies
├── Readme.md               # Project documentation
├── .gitignore              # Git ignore rules
├── train.csv               # Training data (if applicable)
├── gaussian_filtered_images/
│   ├── model.h5            # Pre-trained Keras model
│   ├── binary_diabetic_retinopathy_model.h5
│   └── export.pkl          # (Optional) Model artifacts
├── Unwanted UI/            # Alternate/legacy UI scripts
│   ├── streamlit_app_v2.py
│   └── streamlit_app_working.py
└── ...                     # Other files and folders
```

## Setup Instructions

### 1. Clone the Repository
```sh
git clone <your-repo-url>
cd Diabetic_Retinopathy_Detection
```

### 2. Create and Activate a Virtual Environment (Recommended)
```sh
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```sh
pip install -r requirements.txt
```

### 4. Start the Flask Backend
```sh
python app.py
```
The Flask server will start (by default on http://127.0.0.1:5000).

### 5. Start the Streamlit Frontend
In a new terminal (with the virtual environment activated):
```sh
streamlit run streamlit_app.py
```
The app will be available at http://localhost:8501.

## Usage
1. Open the Streamlit app in your browser.
2. Upload a retinal image.
3. View the prediction results and model confidence.

## Notes
- The Flask backend must be running for the Streamlit app to function.
- Model files are included in the `gaussian_filtered_images/` directory.
- The `Unwanted UI/` folder contains alternate or legacy UI scripts and is not required for main usage.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is for educational and research purposes. Please check with the repository owner for licensing details.
```
Note that you should replace `https://github.com/username/diabetic-retinopathy-detection/blob/main/conda_guide.md` with the actual link to your Conda guide.
# Diabetic-Retinopathy-Detection
