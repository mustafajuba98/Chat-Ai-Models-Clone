from flask import Flask, render_template, request, jsonify, send_from_directory, session
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import os
import requests
import logging
from datetime import datetime
import uuid
import base64
import mimetypes
import PyPDF2
import docx
import csv
from io import BytesIO
from PIL import Image

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
load_dotenv()
app.secret_key = os.urandom(24)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'doc', 'docx', 'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# API Keys

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

# Ai Models Added
AVAILABLE_MODELS = {
    'gpt-4o-mini': {
        'name': 'GPT-4o Mini',
        'supports_vision': True,
        'api_type': 'openai',
        'api_model': 'gpt-4o-mini'
    },
    'claude-3-opus': {
        'name': 'Claude 3 Opus',
        'supports_vision': True,
        'api_type': 'anthropic',
        'api_model': 'claude-3-opus-20240229'
    },
    'claude-3-sonnet': {
        'name': 'Claude 3 Sonnet',
        'supports_vision': True,
        'api_type': 'anthropic',
        'api_model': 'claude-3-sonnet-20240229'
    },
    'dall-e-2': {
        'name': 'DALL-E 2',
        'supports_vision': False,
        'is_image_gen': True,
        'api_type': 'openai',
        'api_model': 'dall-e-2'
    },
    'dall-e-3': {
        'name': 'DALL-E 3',
        'supports_vision': False,
        'is_image_gen': True,
        'api_type': 'openai',
        'api_model': 'dall-e-3'
    }
}

# Storing chat sessions in memory for simplicity
chat_sessions = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_file(file_path):
    file_extension = file_path.split('.')[-1].lower()
    try:
        if file_extension in ['jpg', 'jpeg', 'png', 'gif']:
            return None
        elif file_extension == 'pdf':
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    text += pdf_reader.pages[page_num].extract_text() + "\n"
            return text
        elif file_extension in ['doc', 'docx']:
            doc = docx.Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        elif file_extension == 'csv':
            text = []
            with open(file_path, 'r') as file:
                csv_reader = csv.reader(file)
                for row in csv_reader:
                    text.append(','.join(row))
            return "\n".join(text)
        elif file_extension == 'txt':
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        else:
            return "Unsupported file format"
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {str(e)}")
        return f"Error processing file: {str(e)}"
# image resizing function if the image is too large to send
def resize_image_if_needed(image_path, max_size=10*1024*1024):
    """image resizing function if the image is too large to send"""
    file_size = os.path.getsize(image_path)
    if file_size <= max_size:
        return image_path
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            ratio = (max_size / file_size) ** 0.5
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            resized_img = img.resize((new_width, new_height))
            output = BytesIO()
            fmt = img.format if img.format else 'JPEG'
            resized_img.save(output, format=fmt, quality=85)
            resized_path = f"{image_path}_resized"
            with open(resized_path, 'wb') as f:
                f.write(output.getvalue())
            return resized_path
    except Exception as e:
        logger.error(f"Error resizing image: {str(e)}")
        return image_path

def call_openai_api(messages, model, max_tokens=1000):
    """ open ai api calling  """
    if not OPENAI_API_KEY:
        return "OpenAI API key not configured", 500
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7
    }
    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"], 200
    except requests.exceptions.RequestException as e:
        logger.error(f"OpenAI API error: {str(e)}")
        if response := getattr(e, 'response', None):
            logger.error(f"Response: {response.text}")
            return f"Failed to get response from OpenAI: {response.text}", 500
        return f"Failed to get response from OpenAI: {str(e)}", 500

def call_anthropic_api(messages, model, max_tokens=1000):
    """استدعاء واجهة Anthropic للدردشة"""
    if not ANTHROPIC_API_KEY:
        return "Anthropic API key not configured", 500
    headers = {
        "Content-Type": "application/json",
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01"
    }
    system_message = "You are Claude, a helpful AI assistant."
    human_message_parts = []
    for message in messages:
        if message["role"] == "system":
            system_message = message["content"]
        elif message["role"] == "user":
            if isinstance(message["content"], list):
                text_content = ""
                image_content = []
                for content_part in message["content"]:
                    if content_part["type"] == "text":
                        text_content += content_part["text"]
                    elif content_part["type"] == "image_url":
                        image_url = content_part["image_url"]["url"]
                        if image_url.startswith("data:"):
                            image_content.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": image_url.split(';')[0].split(':')[1],
                                    "data": image_url.split(',')[1]
                                }
                            })
                human_message_parts.append({"type": "text", "text": text_content})
                human_message_parts.extend(image_content)
            else:
                human_message_parts.append({"type": "text", "text": message["content"]})
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_message,
        "messages": [
            {
                "role": "user",
                "content": human_message_parts
            }
        ]
    }
    try:
        response = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["content"][0]["text"], 200
    except requests.exceptions.RequestException as e:
        logger.error(f"Anthropic API error: {str(e)}")
        if response := getattr(e, 'response', None):
            logger.error(f"Response: {response.text}")
            return f"Failed to get response from Anthropic: {response.text}", 500
        return f"Failed to get response from Anthropic: {str(e)}", 500

def generate_image_openai(prompt, model="dall-e-3", size="1024x1024"):
    """توليد صورة باستخدام OpenAI's DALL-E API مع تسجيل تفاصيل الاستجابة للتصحيح"""
    if not OPENAI_API_KEY:
        return None, "OpenAI API key not configured"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    payload = {
        "model": model,
        "prompt": prompt,
        "n": 1,
        "size": size
    }
    try:
        response = requests.post("https://api.openai.com/v1/images/generations", headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        logger.debug(f"Image generation response: {data}")
        return data["data"][0]["url"], None
    except requests.exceptions.RequestException as e:
        logger.error(f"Image generation error: {str(e)}")
        if response := getattr(e, 'response', None):
            logger.error(f"Response text: {response.text}")
        error_message = f"Failed to generate image: {str(e)}"
        return None, error_message

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat/<chat_id>', methods=['POST'])
def chat(chat_id):
    try:
        if chat_id not in chat_sessions:
            return jsonify({"error": "Chat session not found"}), 404
        data = request.form if request.form else request.get_json()
        user_message = data.get('message', '')
        selected_model = data.get('model', 'gpt-4o-mini')
        if selected_model not in AVAILABLE_MODELS:
            selected_model = 'gpt-4o-mini'
        model_info = AVAILABLE_MODELS[selected_model]
        messages = []
        messages.append({
            "role": "system", 
            "content": "You are a helpful assistant that can discuss various topics and analyze images when provided."
        })
        for m in chat_sessions[chat_id]['messages']:
            if m["role"] == "user" and "image" in m:
                if model_info['supports_vision']:
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": m["content"]},
                            {"type": "image_url", "image_url": {"url": m["image"]}}
                        ]
                    })
                else:
                    messages.append({"role": m["role"], "content": m["content"] + " [Note: Image was attached but cannot be processed by this model]"})
            else:
                messages.append({"role": m["role"], "content": m["content"]})
        file_message = None

        if model_info.get('is_image_gen', False):
            image_url, error = generate_image_openai(user_message, model=model_info['api_model'])
            if error:
                return jsonify({"error": error}), 500
            chat_sessions[chat_id]['messages'].append({
                "role": "user",
                "content": f"Generate image: {user_message}",
                "timestamp": datetime.now().strftime("%H:%M")
            })
            chat_sessions[chat_id]['messages'].append({
                "role": "assistant",
                "content": "Here's the generated image:",
                "image": image_url,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            return jsonify({
                "response": "Here's the generated image:",
                "image_url": image_url,
                "status": "success",
            })

        if request.files and 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                mime_type = mimetypes.guess_type(file_path)[0]
                if mime_type and mime_type.startswith('image/'):
                    file_path = resize_image_if_needed(file_path)
                    with open(file_path, 'rb') as img_file:
                        img_data = base64.b64encode(img_file.read()).decode('utf-8')
                    image_url = f"data:{mime_type};base64,{img_data}"
                    chat_sessions[chat_id]['messages'].append({
                        "role": "user",
                        "content": user_message,
                        "image": image_url,
                        "timestamp": datetime.now().strftime("%H:%M")
                    })
                    if model_info['supports_vision']:
                        messages.append({
                            "role": "user",
                            "content": [
                                {"type": "text", "text": user_message},
                                {"type": "image_url", "image_url": {"url": image_url}}
                            ]
                        })
                    else:
                        image_notice = "[Note: An image was attached but cannot be processed by this model]"
                        messages.append({
                            "role": "user",
                            "content": f"{user_message} {image_notice}"
                        })
                    file_message = {
                        "type": "image",
                        "filename": filename
                    }
                else:
                    extracted_text = extract_text_from_file(file_path)
                    if extracted_text:
                        file_content = f"File content from {filename}:\n{extracted_text}"
                        combined_message = f"{user_message}\n\n{file_content}"
                        chat_sessions[chat_id]['messages'].append({
                            "role": "user",
                            "content": combined_message,
                            "timestamp": datetime.now().strftime("%H:%M")
                        })
                        messages.append({
                            "role": "user",
                            "content": combined_message
                        })
                        file_message = {
                            "type": "document",
                            "filename": filename
                        }
            else:
                return jsonify({"error": "Invalid file"}), 400
        elif user_message:
            chat_sessions[chat_id]['messages'].append({
                "role": "user",
                "content": user_message,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            messages.append({
                "role": "user",
                "content": user_message
            })
        else:
            return jsonify({"error": "No message or file provided"}), 400

        api_type = model_info['api_type']
        model = model_info['api_model']
        if api_type == 'openai':
            ai_message, status_code = call_openai_api(messages, model)
            if status_code != 200:
                return jsonify({"error": ai_message}), status_code
        elif api_type == 'anthropic':
            ai_message, status_code = call_anthropic_api(messages, model)
            if status_code != 200:
                return jsonify({"error": ai_message}), status_code
        else:
            return jsonify({"error": "Unsupported API type"}), 400

        chat_sessions[chat_id]['messages'].append({
            "role": "assistant",
            "content": ai_message,
            "timestamp": datetime.now().strftime("%H:%M")
        })

        return jsonify({
            "response": ai_message,
            "status": "success",
            "file": file_message
        })

    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/regenerate/<chat_id>', methods=['POST'])
def regenerate_response(chat_id):
    try:
        if chat_id not in chat_sessions:
            return jsonify({"error": "Chat session not found"}), 404
        data = request.get_json()
        selected_model = data.get('model', 'gpt-4o-mini')
        if selected_model not in AVAILABLE_MODELS:
            selected_model = 'gpt-4o-mini'
        model_info = AVAILABLE_MODELS[selected_model]
        chat_history = chat_sessions[chat_id]['messages']
        if chat_history and chat_history[-1]['role'] == 'assistant':
            chat_history.pop()
        messages = []
        messages.append({
            "role": "system", 
            "content": "You are a helpful assistant that can discuss various topics and analyze images when provided."
        })
        for m in chat_history:
            if m["role"] == "user" and "image" in m:
                if model_info['supports_vision']:
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": m["content"]},
                            {"type": "image_url", "image_url": {"url": m["image"]}}
                        ]
                    })
                else:
                    messages.append({"role": m["role"], "content": m["content"] + " [Note: Image was attached but cannot be processed by this model]"})
            else:
                messages.append({"role": m["role"], "content": m["content"]})
        api_type = model_info['api_type']
        model = model_info['api_model']
        if api_type == 'openai':
            ai_message, status_code = call_openai_api(messages, model)
            if status_code != 200:
                return jsonify({"error": ai_message}), status_code
        elif api_type == 'anthropic':
            ai_message, status_code = call_anthropic_api(messages, model)
            if status_code != 200:
                return jsonify({"error": ai_message}), status_code
        else:
            return jsonify({"error": "Unsupported API type"}), 400
        chat_sessions[chat_id]['messages'].append({
            "role": "assistant",
            "content": ai_message,
            "timestamp": datetime.now().strftime("%H:%M")
        })
        return jsonify({
            "response": ai_message,
            "status": "success"
        })
    except Exception as e:
        logger.error(f"Regenerate error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/new_chat', methods=['POST'])
def new_chat():
    chat_id = str(uuid.uuid4())
    chat_sessions[chat_id] = {
        'messages': [],
        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'title': 'New Chat'
    }
    return jsonify({"chat_id": chat_id})

@app.route('/get_chats', methods=['GET'])
def get_chats():
    return jsonify(chat_sessions)

@app.route('/get_models', methods=['GET'])
def get_models():
    return jsonify(AVAILABLE_MODELS)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
