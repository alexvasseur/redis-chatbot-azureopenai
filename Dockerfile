FROM python:3.9.10-slim-buster

RUN apt-get update && apt-get install python-tk python3-tk tk-dev git -y

WORKDIR /app

COPY app/requirements.txt .

RUN python -m pip install --upgrade pip

# Alternative way to avoid downloading 10GB of GPU dependencies
# from sentence-transformers
#RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
#RUN pip install transformers tqdm numpy scikit-learn scipy nltk sentencepiece
#RUN pip install --no-deps sentence-transformers

RUN pip install -r requirements.txt

EXPOSE 8080

CMD ["streamlit", "run", "main.py", "--server.port", "8080", "--server.enableXsrfProtection", "false", "--browser.gatherUsageStats", "false"]