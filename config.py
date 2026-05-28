import os

# Base directory of project
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Upload folder path
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

# Allowed file extensions
ALLOWED_EXTENSIONS = {"pdf", "txt"}

# Maximum upload size (16 MB)
MAX_CONTENT_LENGTH = 16 * 1024 * 1024

# Secret key for Flask security
SECRET_KEY = "document_summarizer_secret_key"