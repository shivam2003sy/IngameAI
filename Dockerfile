# Base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

RUN pip install -q gdown inference-gpu \
    && pip install -q onnxruntime-gpu==1.18.0 --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ \
    && pip install ultralytics \
    && pip uninstall -y supervision \
    && pip install -q supervision>=0.23.0 \
    && pip install streamlit \
    && pip install umap-learn \
    && pip install sentencepiece \
    && pip install tqdm


# Copy the application code into the container
COPY . .

# Expose the Streamlit default port
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
