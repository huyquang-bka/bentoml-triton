from nvcr.io/nvidia/tritonserver:22.12-py3
RUN apt-get update && apt-get install -y pkg-config libhdf5-dev ffmpeg libsm6 libxext6
RUN pip3 install --upgrade pip
COPY requirements.txt .
RUN pip3 install -r requirements.txt
CMD bash