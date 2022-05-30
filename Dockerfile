FROM nvcr.io/nvidia/tensorflow:19.10-py3

RUN pip install --upgrade pip
RUN pip install imageio imageio-ffmpeg h5py opencv-python
RUN pip install matplotlib keras==2.2.5 requests

RUN apt-get update -y
RUN apt-get install ffmpeg libsm6 libxext6 -y

# WORKDIR /workspace/
# COPY . .

ENTRYPOINT [ "python", "deepGenPro.py" ]

# docker build --rm -t dgp .
# docker run --name DGP --gpus all --rm -it -v <absolute_local_path>:/workspace/ dgp   --cloth_dir=./data/model_1/cloth_1/ --model_dir=./data/model_1/model_info/ --cloth_sleeve=short --output_dir=./output/