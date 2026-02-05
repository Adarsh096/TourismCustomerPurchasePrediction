import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import mlflow

# --- UPDATED MLFLOW CONFIG ---
HUGGINGFACE_USER_NAME = os.getenv('HUGGINGFACE_USER_NAME')
MLFLOW_SPACE_NAME = "tourism-mlflow-tracking-server"

# Using the direct API URL for the space
# Format: https://<user>-<space_name>.hf.space
tracking_uri = f"https://{HUGGINGFACE_USER_NAME}-{MLFLOW_SPACE_NAME}.hf.space"
print("MLFlow tracking URL: ", tracking_uri)

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("tourism_project-training-experiment")

HUGGINGFACE_DATASET_NAME = os.getenv('HUGGINGFACE_DATASET_NAME')
HUGGINGFACE_MODEL_NAME = os.getenv('HUGGINGFACE_MODEL_NAME')
api = HfApi()

repo_id = f"{HUGGINGFACE_USER_NAME}/{HUGGINGFACE_DATASET_NAME}"
Xtrain = pd.read_csv(f"hf://datasets/{repo_id}/Xtrain.csv")
Xtest = pd.read_csv(f"hf://datasets/{repo_id}/Xtest.csv")
ytrain = pd.read_csv(f"hf://datasets/{repo_id}/ytrain.csv")
ytest = pd.read_csv(f"hf://datasets/{repo_id}/ytest.csv")

class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Splitting numeric list into two lists, one which are to be scale and another not to be scaled as they are categorical/ordinal values.
# Columns that need Scaling (Continuous)
numeric_scaling = [
    'Age', 'DurationOfPitch', 'NumberOfPersonVisiting',
    'NumberOfFollowups', 'NumberOfTrips', 'NumberOfChildrenVisiting', 'MonthlyIncome'
]

# Columns to keep as-is (Numerical but categorical/binary in nature)
numeric_passthrough = [
    'CityTier', 'PreferredPropertyStar', 'Passport',
    'OwnCar', 'PitchSatisfactionScore'
]

# categorical columns:
categorical_features = [
    'TypeofContact',  'Occupation',
    'Gender',  'ProductPitched',
    'MaritalStatus',  'Designation'
]

# Updating the ColumnTransformer
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_scaling),        # Apply scaling here
    (OneHotEncoder(handle_unknown='ignore'), categorical_features), # Encoding
    ("passthrough", numeric_passthrough)       # Keep these exactly as they are
)

xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)
param_grid = {
    'xgbclassifier__n_estimators': [50, 100],
    'xgbclassifier__max_depth': [3, 4],
    'xgbclassifier__learning_rate': [0.05, 0.1],
}

model_pipeline = make_pipeline(preprocessor, xgb_model)

with mlflow.start_run():
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Logging best parameters
    mlflow.log_params(grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # Evaluation
    test_report = classification_report(ytest, best_model.predict(Xtest), output_dict=True)
    mlflow.log_metrics({
        "test_accuracy": test_report['accuracy'],
        "test_f1-score": test_report['1']['f1-score'],
        "test_precision-score": test_report['1']['precision'],
        "test_recall-score": test_report['1']['recall']
    })

    # Save and Log Artifact in MLFlow
    model_path = "best_model_v1.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")

    # Upload to HF Model Hub
    model_repo_id = f"{HUGGINGFACE_USER_NAME}/{HUGGINGFACE_MODEL_NAME}"
    try:
        api.repo_info(repo_id=model_repo_id, repo_type="model")
    except RepositoryNotFoundError:
        create_repo(repo_id=model_repo_id, repo_type="model", private=False)

    api.upload_file(
        path_or_fileobj=model_path, #src name
        path_in_repo="model.joblib", #target name
        repo_id=model_repo_id,
        repo_type="model",
    )
