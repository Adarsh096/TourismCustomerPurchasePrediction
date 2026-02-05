from huggingface_hub import HfApi
import os

#common constants:
HUGGINGFACE_USER_NAME = os.getenv('HUGGINGFACE_USER_NAME')
HUGGINGFACE_SPACE_NAME = os.getenv('HUGGINGFACE_SPACE_NAME')

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="tourism_project/deployment",
    repo_id=f"{HUGGINGFACE_USER_NAME}/{HUGGINGFACE_SPACE_NAME}",  # the target repo
    repo_type="space", # dataset, model, or space
    path_in_repo="", # optional: subfolder path inside the repo
)
