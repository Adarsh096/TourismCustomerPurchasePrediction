# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

#common constants:
HUGGINGFACE_DATASET_NAME = os.getenv('HUGGINGFACE_DATASET_NAME')
HUGGINGFACE_USER_NAME = os.getenv('HUGGINGFACE_USER_NAME')

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = f"hf://datasets/{HUGGINGFACE_USER_NAME}/{HUGGINGFACE_DATASET_NAME}/tourism.csv"
tourism_dataset = pd.read_csv(DATASET_PATH, index_col=0)
print("Dataset loaded successfully.")

# Define the target variable for the classification task
target = 'ProdTaken'

# List of numerical features in the dataset
numeric_features = [
    'Age',
    'CityTier',
    'DurationOfPitch',
    'NumberOfPersonVisiting',
    'NumberOfFollowups',
    'PreferredPropertyStar',
    'NumberOfTrips',
    'Passport',
    'PitchSatisfactionScore',
    'OwnCar',
    'NumberOfChildrenVisiting',
    'MonthlyIncome'

]

# List of categorical features in the dataset
categorical_features = [
    'TypeofContact',
    'Occupation',
    'Gender',
    'ProductPitched',
    'MaritalStatus',
    'Designation'
]

# Improper value treatment:
# Female gender values are recorded incorrectly:
tourism_dataset['Gender'] = tourism_dataset['Gender'].replace('Fe Male', 'Female')
# Single and Unmarried mean the same thing:
tourism_dataset['MaritalStatus'] = tourism_dataset['MaritalStatus'].replace('Unmarried', 'Single')

# Define predictor matrix (X) using selected numeric and categorical features
X = tourism_dataset[numeric_features + categorical_features]

# Define target variable
y = tourism_dataset[target]


# Split dataset into train and test
# Split the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.2,     # 20% of the data is reserved for testing
    random_state=42    # Ensures reproducibility by setting a fixed random seed
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id=f"{HUGGINGFACE_USER_NAME}/{HUGGINGFACE_DATASET_NAME}",
        repo_type="dataset",
    )
