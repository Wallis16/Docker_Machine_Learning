# init a base image
FROM ubuntu:22.04

# define the present working directory
WORKDIR /App

# copy the contents into the working dir
ADD . /App

# install pip
RUN apt-get update && apt-get install -y \
    python3-pip

# install dependencies
RUN pip install -r requirements.txt
COPY . /App

# 
ENTRYPOINT [ "python3" ]

# command to start the container
CMD ["app_training.py"]