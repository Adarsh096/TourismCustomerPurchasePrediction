import os
import time
from huggingface_hub import HfApi, create_repo, get_space_runtime

def setup_and_wait():
    api = HfApi()
    repo_id = f"{os.getenv('HUGGINGFACE_USER_NAME')}/tourism-mlflow-tracking-server"
    print(f"repo_id: {repo_id}")

    # 1. Create/Update Repo
    try:
        create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker")
    except Exception:
        print(f"Repo: {repo_id} already exists.")

    # 2. Upload Dockerfile
    api.upload_file(
        path_or_fileobj="tourism_project/model_building/mlflow/Dockerfile",
        path_in_repo="Dockerfile",
        repo_id=repo_id,
        repo_type="space"
    )

    # 3. Wait for Healthy Status (Polled via HF API instead of Curl)
    print("Waiting for MLflow to pass Healthcheck...")
    for _ in range(20):  # 5 minutes max (15s * 20)
        status = get_space_runtime(repo_id=repo_id).stage
        if status == "RUNNING":
            print("MLflow is Live!")
            return
        if "ERROR" in status:
            raise Exception(f"Space failed with status: {status}")
        time.sleep(15)#seconds
    raise TimeoutError("MLflow took too long to start.")

if __name__ == "__main__":
    setup_and_wait()
