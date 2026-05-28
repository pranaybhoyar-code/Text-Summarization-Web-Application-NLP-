from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

import os

# Configurations
from config import *

# NLP Modules
from nlp.extract_text import extract_text_from_file
from nlp.preprocess import preprocess_text
from nlp.transformer_summarizer import (
    bart_summarize,
    bert_summarize
)

# =========================
# INITIALIZE FLASK APP
# =========================

app = Flask(__name__)

# =========================
# LOAD CONFIGURATIONS
# =========================

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

app.secret_key = SECRET_KEY

# Create uploads folder automatically
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# =========================
# FILE VALIDATION FUNCTION
# =========================

def allowed_file(filename):

    return "." in filename and \
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# =========================
# HOME ROUTE
# =========================

@app.route("/", methods=["GET", "POST"])

def index():

    summary = ""
    original_text = ""

    # Run only when form submitted
    if request.method == "POST":
        print("Form submitted")

        try:

            # =========================
            # GET FORM DATA
            # =========================

            original_text = request.form.get("text", "")

            model_choice = request.form.get("model", "bart")

            file = request.files.get("file")

            # =========================
            # FILE UPLOAD HANDLING
            # =========================

            if file and file.filename != "":

                # Validate file type
                if allowed_file(file.filename):

                    # Secure file name
                    filename = secure_filename(
                        file.filename
                    )

                    # Create full file path
                    filepath = os.path.join(
                        app.config["UPLOAD_FOLDER"],
                        filename
                    )

                    # Save uploaded file
                    file.save(filepath)

                    # Extract text from document
                    original_text = extract_text_from_file(
                        filepath
                    )

                else:

                    return render_template(
                        "index.html",
                        error="Only PDF and TXT files are allowed."
                    )

            # =========================
            # VALIDATE INPUT TEXT
            # =========================

            if original_text and len(original_text.strip()) > 0:

                # =========================
                # PREPROCESS TEXT
                # =========================

                cleaned_text = preprocess_text(
                    original_text
                )
                print("Preprocessing completed")

                # =========================
                # GENERATE SUMMARY
                # =========================

                if model_choice == "bart":

                    summary = bart_summarize(
                        cleaned_text
                    )
                    print("BART summarization completed")

                elif model_choice == "bert":

                    summary = bert_summarize(
                        cleaned_text
                    )
                    print("BERT summarization completed")

                else:

                    summary = "Invalid model selected."

                # =========================
                # SHOW RESULT PAGE
                # =========================

                return render_template(
                    "result.html",
                    summary=summary,
                    text=original_text
                )

            else:

                return render_template(
                    "index.html",
                    error="Please enter text or upload a document."
                )

        except Exception as e:

            return render_template(
                "index.html",
                error=f"Error: {str(e)}"
            )

    # =========================
    # LOAD HOME PAGE
    # =========================

    return render_template("index.html")


# =========================
# RUN FLASK SERVER
# =========================

if __name__ == "__main__":

    app.run(debug=True)