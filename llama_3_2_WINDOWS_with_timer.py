import time
import torch

print(torch.version.cuda)

# GPU 사용 여부 확인
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    device = torch.device("cuda")
else:
    print("GPU is not available, using CPU.")
    device = torch.device("cpu")

from transformers import MllamaForConditionalGeneration, AutoProcessor
from PIL import Image
import requests

# Update the paths to use relative paths with Assets directory
model_path = "./Assets/Models"
image_path = "./Assets/Images/mathProblem2.png"

# Initialize model and processor
model = MllamaForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    # device_map="cuda"  # Uncomment this line if you want to force all parts of the model to be on GPU
)
processor = AutoProcessor.from_pretrained(model_path)

# Open the image
image = Image.open(image_path)

# Create a message data structure
messageDataStructure = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {
                "type": "text",
                "text": "Compute the side of c of the triangle in the image.",
            },
        ],
    }
]

# Create text input
textInput = processor.apply_chat_template(
    messageDataStructure, add_generation_prompt=True
)

# Call the processor
inputs = processor(image, textInput, return_tensors="pt").to(device)

# 시작 시간 기록
start_time = time.time()

# Here, change the number of tokens to get a more detailed answer
output = model.generate(**inputs, max_new_tokens=2000)

# Here, we decode and store the response so we can print it
generatedOutput = processor.decode(output[0])

# 종료 시간 기록
end_time = time.time()

# 소요 시간 계산 및 출력
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")

print(generatedOutput)

with open("output.txt", "w", encoding="utf-8") as text_file:
    text_file.write(generatedOutput)
