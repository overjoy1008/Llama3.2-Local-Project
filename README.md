# Llama3.2 Local Project
- This project is compatible with MAC & WINDOWS

## 1. How to activate the Project
- This project is based on `Python 3.12`, so other versions are NOT guaranteed to work
- You must create venv and install packages/libraries from requirements.txt
- You must create `.env` file for Hugging Face Token

## 2. How to activate .venv on VS Code
- Click `Show and Run Commands` at the top input field.
- Type 'env' to find `Python: Create Environment...`
- Click `Venv` and create using `Python 3.12`
- Make sure to select dependency `requirements.txt` to install (Image Below)
  <img width="609" alt="스크린샷 2024-11-09 오후 1 03 35" src="https://github.com/user-attachments/assets/1ac42b53-3124-401b-ba23-0e9c81b77805">
- For MAC users, type in the terminal `source .venv/bin/activate` to enter venv, and `deactivate` to exit.
- For WINDOWS users, type in the terminal `.venv\Scripts\activate` to enter venv, and `deactivate` to exit.

## 3. How to Manually install from requirements.txt / Download new packages
- Make sure you activated venv (Image Below)
  - MAC:
    
    <img width="1161" alt="스크린샷 2024-11-09 오후 1 24 18" src="https://github.com/user-attachments/assets/3420a9df-af67-4077-98fc-1830d4f6decd">
  - WINDOWS:

    <img width="685" alt="스크린샷 2024-11-09 오후 1 23 47" src="https://github.com/user-attachments/assets/4ac3b185-14c9-4930-ac08-c9d330fd1fea">
- `pip install -r requirements.txt` will let you install all the packages in the requirements.txt
- `pip freeze > requirements.txt` will update the newly downloaded package to requirements.txt
- For WINDOWS users, you must use Pytorch CUDA by:
  1. `pip uninstall torch torchvision torchaudio`
  2. `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124` (*You may need to have c++ installed in order to install pytorch*)

## 4. Using Dotenv for Security
- Your .gitignore should look like this:
  ```
  # Environment Variables
  .env

  # Python
  __pycache__/
  *.py[cod]
  *$py.class
  .Python
  venv/

  # VS Code
  .vscode/
  *.code-workspace
  .history/
  
  # OS
  .DS_Store
  Thumbs.db
  ```
- Create .env file and place your tokens/API keys
  - For example, `HUGGINGFACE_LLAMA_3_2_TOKEN = "PLACE YOUR TOKEN HERE"`
- **MAKE SURE `.env` TURNS GREY!! OR ELSE IT WON'T WORK (Image Below)**
  
  <img width="156" alt="스크린샷 2024-11-09 오후 1 23 13" src="https://github.com/user-attachments/assets/b4833dda-993a-474b-bf10-ac0711aa2f37">
- In case your .env doesn't turn grey:
  - Download dotenv by typing `pip install python-dotenv`
  - Check your .gitignore and make sure it includes `.env`
