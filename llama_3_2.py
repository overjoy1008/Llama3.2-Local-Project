import torch
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
            {"type": "text", "text": "Compute the side of c of the triangle in the image."},
        ],
    }
]

# Create text input
textInput = processor.apply_chat_template(
    messageDataStructure, add_generation_prompt=True
)

# Call the processor
inputs = processor(image, textInput, return_tensors="pt").to(model.device)

# Here, change the number of tokens to get a more detailed answer
output = model.generate(**inputs, max_new_tokens=2000)

# Here, we decode and store the response so we can print it
generatedOutput = processor.decode(output[0])
