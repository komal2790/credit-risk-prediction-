Credit Risk Prediction Using Machine Learning

Project Overview:
This project focuses on predicting the credit risk of individuals—categorized as either "good" or "bad"—based on the German Credit Dataset. The goal is to assist financial institutions in evaluating loan applications using machine learning techniques.

Data Preprocessing:

Loaded and explored the German Credit Dataset.

Created a custom target variable credit_risk using the condition: if Credit amount > 5000 and Duration > 20, then labeled as "good", else "bad".

Applied feature scaling and one-hot encoding to prepare the data for modeling.

The dataset was imbalanced (847 bad, 153 good), so SMOTE (Synthetic Minority Oversampling Technique) was used to balance the classes.

Machine Learning Models Used:
The dataset was split into training and testing sets in a 70:30 ratio. The following models were trained and evaluated:

Logistic Regression: Served as the baseline model and achieved around 95% accuracy.

Random Forest Classifier: Delivered the best overall performance with strong precision and recall.

K-Nearest Neighbors (KNN): Provided decent results; performance varied based on the value of K.

Decision Tree: Easy to interpret but showed signs of overfitting.

Support Vector Machine (SVM): Worked well on scaled data; performance depended on kernel selection and tuning.

Model Evaluation Metrics:

Accuracy

Confusion Matrix

Precision, Recall, and F1-Score

Sample results:
Accuracy: 95.3%
Confusion Matrix:
Predicted vs Actual
[[256 3]
[11 30]]

Classification Report:
Class 0 (bad): Precision 0.96, Recall 0.99
Class 1 (good): Precision 0.91, Recall 0.73

Streamlit Web App:
A Streamlit-based user interface was created for real-time predictions. The web app allows users to input features like age, credit amount, duration, housing, and job details, and then predicts whether the credit risk is good or bad using the trained model (Random Forest).

To run the app locally:
Use the command streamlit run app.py in your terminal.

Folder Structure:

data/: Contains the dataset

models/: Stores trained models (optional)

app.py: Streamlit web application

notebooks/: Contains Jupyter notebooks used for training and evaluation

README.md: Project documentation

Tech Stack:

Python

Pandas, NumPy

Scikit-learn

Imbalanced-learn (for SMOTE)

Streamlit (for UI)

Future Improvements:

Hyperparameter tuning using GridSearchCV or RandomizedSearchCV

Model saving using joblib or pickle

Adding interpretability tools like SHAP or LIME

Deploying the app online via Streamlit Cloud or other hosting platforms
