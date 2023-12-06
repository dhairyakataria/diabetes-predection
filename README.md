
# Diabetes Prediction Web App

This project aims to predict the likelihood of diabetes in individuals based on various health parameters using machine learning algorithms.

## Overview

The Diabetes Prediction Web App provides a user-friendly interface to predict the probability of an individual having diabetes. The application utilizes a Random Forest Classifier trained on health data to generate predictions.

### Features

- **Prediction Interface:** Enter health parameters to receive predictions.
- **Data Preprocessing:** Handles missing values and prepares the dataset.
- **Model Training:** Trains a Random Forest Classifier using the preprocessed data.
- **Web Interface:** Built with Flask, allowing easy access via a web browser.

## Setup

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/your-username/diabetes-prediction.git
    cd diabetes-prediction
    ```

2. **Install Dependencies:**

    Ensure you have Python installed. Install the required libraries using pip:

    ```bash
    pip install flask pandas numpy scikit-learn
    ```

3. **Run the Application:**

    Execute the Flask app:

    ```bash
    python app.py
    ```

4. **Access the Application:**

    Open your web browser and navigate to `http://localhost:5000` to access the application.

## Usage

1. **Home Page:**

    The home page provides a form to input health parameters.

2. **Prediction:**

    Fill in the details and click "Predict" to obtain the prediction for diabetes likelihood.

## Files Included

- `app.py`: Contains the Flask web application code, handling user requests and displaying predictions.
- `model.py`: Prepares the dataset, trains the machine learning model, and saves it using pickle for later use.

## Dependencies

- **Flask:** Web framework for creating the application.
- **Pandas:** Data manipulation library for handling datasets.
- **NumPy:** Library for numerical operations.
- **scikit-learn:** Machine learning library for model building and predictions.

## Additional Notes

- The model uses a Random Forest Classifier to predict diabetes likelihood.
- Health parameters include BMI, age, blood pressure, and other relevant factors.
- The application preprocesses data to handle missing values before generating predictions.
