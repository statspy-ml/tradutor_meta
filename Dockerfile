# Start from an NVIDIA CUDA image with Python
FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04

# Install Python and pip
RUN apt-get update && apt-get install -y python3-pip git

# Install PyTorch compatible with CUDA 11.8 (update if needed)
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

# Install transformers and other dependencies
RUN pip3 install git+https://github.com/huggingface/transformers.git sentencepiece protobuf

# Install Jupyter
RUN pip3 install jupyter

# Expose the port Jupyter will run on
EXPOSE 8888

# Set the working directory
WORKDIR /workspace

# Start Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]

