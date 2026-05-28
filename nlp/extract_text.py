import PyPDF2


def extract_text_from_file(filepath):

    text = ""

    # PDF File Handling
    if filepath.endswith(".pdf"):

        try:
            with open(filepath, "rb") as pdf_file:

                pdf_reader = PyPDF2.PdfReader(pdf_file)

                for page in pdf_reader.pages:

                    extracted_text = page.extract_text()

                    if extracted_text:
                        text += extracted_text + "\n"

        except Exception as e:
            raise Exception(f"Error reading PDF file: {str(e)}")

    # TXT File Handling
    elif filepath.endswith(".txt"):

        try:
            with open(filepath, "r", encoding="utf-8") as txt_file:

                text = txt_file.read()

        except UnicodeDecodeError:

            # Fallback encoding
            with open(filepath, "r", encoding="latin-1") as txt_file:

                text = txt_file.read()

        except Exception as e:
            raise Exception(f"Error reading TXT file: {str(e)}")

    else:
        raise Exception("Unsupported file format.")

    return text.strip()