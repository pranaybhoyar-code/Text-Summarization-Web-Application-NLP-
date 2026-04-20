from flask import Flask, render_template, request
from model import bart_summarize, bert_summarize
import os
import PyPDF2

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/", methods=["GET", "POST"])
def index():
    summary = ""
    text = ""

    if request.method == "POST":

        text = request.form.get("text")
        model_choice = request.form.get("model")
        file = request.files.get("file")

        if file and file.filename != "":
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            if file.filename.endswith(".pdf"):
                text = ""
                pdf_reader = PyPDF2.PdfReader(filepath)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            else:
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()

        if text:
            if model_choice == "bart":
                summary = bart_summarize(text)
            else:
                summary = bert_summarize(text)

    return render_template("index.html", summary=summary, text=text)


if __name__ == "__main__":
    app.run(debug=True)