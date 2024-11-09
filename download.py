import os
from huggingface_hub import snapshot_download
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# def download_llama_vision():
#     # Create Assets/Models directory if it doesn't exist
#     model_dir = Path("Assets/Models")
#     model_dir.mkdir(parents=True, exist_ok=True)

#     try:
#         # Download the model
#         model_id = "meta-llama/llama-2-7b-chat-hf"  # Replace with actual model ID
#         token = os.getenv("HUGGINGFACE_LLAMA_3_2_TOKEN")
#         snapshot_download(
#             repo_id=model_id,
#             local_dir=str(model_dir),
#             token=token,
#             ignore_patterns=["*.safetensors", "*.msgpack"],
#             local_dir_use_symlinks=False,
#         )
#         print(f"Successfully downloaded model to {model_dir}")

#     except Exception as e:
#         print(f"Error downloading model: {e}")


# if __name__ == "__main__":
#     download_llama_vision()

snapshot_download(
    repo_id="meta-llama/Llama-3.2-11B-Vision-Instruct",
    use_auth_token=os.getenv("HUGGINGFACE_LLAMA_3_2_TOKEN"),
    local_dir="Assets/Models",
)