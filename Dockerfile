FROM ubuntu:20.04

RUN apt-get update
RUN apt-get install wget curl git -y
RUN apt-get install python3 python3-pip -y
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN pip install -U pip
RUN pip install onnx numpy onnx2tf tensorflow -y
RUN pip install onnx_graphsurgeon psutil sng4onnx 