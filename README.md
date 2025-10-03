![farmer](https://github.com/muhammadrehanazam/Crop_prediciton_Model/blob/main/farmer_in_a_field.jpg)

# 🌱 Crop Prediction using Logistic Regression

This project applies **Logistic Regression** to predict suitable crops based on soil nutrient values and pH levels.  
It demonstrates **data preprocessing, feature scaling, model training, evaluation, and feature-wise performance analysis**.  



## 📊 Dataset

The dataset (`soil_measure.csv`) contains soil features:

- **N** → Nitrogen content  
- **P** → Phosphorus content  
- **K** → Potassium content  
- **pH** → Soil pH value  
- **crop** → Target crop label (classification target)  


Dependencies:

pandas

numpy

scikit-learn

matplotlib (optional, for plots)

seaborn (optional, for heatmaps)

🚀 Usage
Run Jupyter Notebook
bash
Copy code
jupyter notebook notebooks/crop_prediction.ipynb
Run Training Script
bash
Copy code
python src/train_model.py

📈 Workflow
Data Loading → Read soil dataset (soil_measure.csv)

Preprocessing → Split into features/target, scale data with StandardScaler

Model Training → Logistic Regression (sklearn.linear_model.LogisticRegression)

Evaluation

F1 Score

Classification Report

Confusion Matrix

Feature importance test (N, P, K, pH separately)

🧪 Results
Example F1 scores per feature (your results may vary):

Feature	F1 Score
N	0.09
P	0.14
K	0.23
pH	0.04

Potassium (K) showed the strongest predictive power.

Logistic Regression struggled to converge with default settings → tuning max_iter is recommended.

⚠️ Notes
F1 scores may differ between environments (e.g., DataCamp vs Jupyter Notebook) due to:

Different scikit-learn versions

Solver defaults (lbfgs vs liblinear)

Convergence behavior

Random splits of training/testing data

To ensure reproducibility:

python
Copy code
log_reg = LogisticRegression(
    solver="lbfgs",
    multi_class="multinomial",
    max_iter=500,
    random_state=42
)
📌 Future Work
Add more soil 
features (temperature, rainfall, etc.)

Try other models (Decision Trees, Random Forest, XGBoost)

