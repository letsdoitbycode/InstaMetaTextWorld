from flask import Flask, request, render_template, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import os
import numpy as np
import base64
import pickle
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications import DenseNet201
from groq import Groq
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)

# Set up upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Load tokenizer and local model
def load_resources():
    """Load the tokenizer, saved captioning model, and DenseNet201 model for feature extraction."""
    with open('models/tokenizer.pkl', 'rb') as file:
        saved_tokenizer = pickle.load(file)

    saved_model = load_model('models/model.keras')

    base_model = DenseNet201()
    fe = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

    return saved_tokenizer, saved_model, fe


# Check if file is an allowed type
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Utility functions
def idx_to_word(integer, tokenizer):
    """Convert an index to a word using the tokenizer."""
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def predict_caption(model, tokenizer, max_length, feature):
    """Generate a caption for an image feature using the local model."""
    in_text = "startseq"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)

        y_pred = model.predict([feature, sequence], verbose=0)
        y_pred = np.argmax(y_pred)

        word = idx_to_word(y_pred, tokenizer)

        if word is None:
            break

        in_text += " " + word

        if word == 'endseq':
            break

    return in_text.replace('startseq', '').replace('endseq', '').strip()


def encode_image(image_file):
    """Encode image to base64."""
    return base64.b64encode(image_file.read()).decode('utf-8')


def generate_caption_groq_cloud(api_key, base64_image):
    """Send a request to Groq Cloud API to generate a caption, hashtags, and a comment for an image."""
    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model="llama-3.2-11b-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """(NO PREAMBLE) Generate a catchy caption for this image that is easy to understand. 
                        Then generate hashtags under the caption to make the post more popular. 
                        Lastly, generate the first comment that I should post as the owner of the image.
                        Format the output in the following way:
                        
                        ### Caption
                        <Caption text here>
                        
                        ### Hashtags
                        <Hashtags here>
                        
                        ### First Comment
                        <Comment here>"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    },
                ],
            }
        ],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=False,
    )

    return completion.choices[0].message.content


def generate_instagram_content_groq(api_key, caption):
    """Use Groq's LLM to generate Instagram hashtags and comments."""
    llm = ChatGroq(
        temperature=0.7,
        groq_api_key=api_key,
        model_name="llama-3.1-70b-versatile"
    )

    prompt = f"(NO PREAMBLE) Generate hashtags under the caption to make the post more popular. \
        Lastly, generate the first comment that I should post as the owner of the image. \
        Format the output in the following way: ### Hashtags <Hashtags here> ### First Comment <Comment here> caption: '{caption}'."

    response = llm.invoke(prompt)
    return response.content


# Load resources once
tokenizer, model, fe = load_resources()


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Process the uploaded image
            img = load_img(file_path, target_size=(224, 224))
            img = img_to_array(img)
            img = img / 255.0
            img = np.expand_dims(img, axis=0)

            # Extract features using DenseNet201
            feature = fe.predict(img, verbose=0)

            # Generate caption locally
            max_length = 74
            local_caption = predict_caption(model, tokenizer, max_length, feature)

            if GROQ_API_KEY:
                # Generate Instagram content using Groq Cloud
                with open(file_path, 'rb') as img_file:
                    base64_image = encode_image(img_file)
                groq_caption = generate_caption_groq_cloud(GROQ_API_KEY, base64_image)
                instagram_content = generate_instagram_content_groq(GROQ_API_KEY, local_caption)
            else:
                groq_caption = "Groq API key not available."
                instagram_content = "No Instagram content available without Groq API key."

            # Return captions and content in JSON format
            return jsonify({
                "local_caption": local_caption,
                "groq_caption": groq_caption,
                "instagram_content": instagram_content
            })

        return jsonify({"error": "Invalid file format"}), 400

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
