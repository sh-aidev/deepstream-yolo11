<div align="center">

# DeepStream YOLOv11

[![NVIDIA](https://img.shields.io/badge/NVIDIA_DeepStream-7.0-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/deepstream-sdk)
[![YOLO](https://img.shields.io/badge/YOLOv11-Custom-orange)](https://github.com/AlexeyAB/darknet)
[![CUDA](https://img.shields.io/badge/CUDA-12.2-green?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)

A customized DeepStream application integrating YOLOv11 for real-time object detection.

</div>

## 📌  Introduction

This repository houses a customized integration of YOLOv11 into the NVIDIA DeepStream SDK for real-time object detection. Based on the [DeepStream YOLO Plugin](https://github.com/marcoslucianops/DeepStream-Yolo.git), this project is adapted to work with the latest YOLOv11 model, providing enhanced detection capabilities with NVIDIA's optimized inference pipeline.

<br>

## 📦  Main Technologies

- [NVIDIA DeepStream SDK](https://developer.nvidia.com/deepstream-sdk) - A complete streaming analytics toolkit for AI-based multi-sensor processing, video, audio, and image understanding.
- [YOLOv11](https://docs.ultralytics.com/models/yolo11/) - Latest iteration of the real-time object detection system, implemented in the Darknet framework.
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) - A parallel computing platform and programming model developed by NVIDIA for general computing on graphical processing units (GPUs).

<br>

## Acknowledgements

This project is an extension of the [DeepStream YOLO](https://github.com/marcoslucianops/DeepStream-Yolo.git), which provides a foundational framework for real-time object detection with YOLO using NVIDIA's DeepStream SDK. For further insights, enhancements, or additional information, please refer to the [original repository](https://github.com/marcoslucianops/DeepStream-Yolo.gits).

We extend our gratitude to the contributors of the original DeepStream YOLO project for their pioneering efforts and for providing a robust platform for further innovation and development.


<br>

## 📁  Project 

The directory structure of the project looks like this:
```
├── LICENSE
├── README.md
├── requirements.txt
└── src
    ├── config_infer_primary_yoloV11.txt
    ├── config_tracker_NvDCF_perf.yml
    ├── deepstream_app_config.txt
    ├── labels.txt
    ├── models
    │   └── .geetkeep
    ├── nvdsinfer_custom_impl_Yolo
    │   ├── calibrator.cpp
    │   ├── calibrator.h
    │   ├── layers
    │   │   ├── activation_layer.cpp
    │   │   ├── activation_layer.h
    │   │   ├── batchnorm_layer.cpp
    │   │   ├── batchnorm_layer.h
    │   │   ├── channels_layer.cpp
    │   │   ├── channels_layer.h
    │   │   ├── convolutional_layer.cpp
    │   │   ├── convolutional_layer.h
    │   │   ├── deconvolutional_layer.cpp
    │   │   ├── deconvolutional_layer.h
    │   │   ├── implicit_layer.cpp
    │   │   ├── implicit_layer.h
    │   │   ├── pooling_layer.cpp
    │   │   ├── pooling_layer.h
    │   │   ├── reorg_layer.cpp
    │   │   ├── reorg_layer.h
    │   │   ├── route_layer.cpp
    │   │   ├── route_layer.h
    │   │   ├── sam_layer.cpp
    │   │   ├── sam_layer.h
    │   │   ├── shortcut_layer.cpp
    │   │   ├── shortcut_layer.h
    │   │   ├── slice_layer.cpp
    │   │   ├── slice_layer.h
    │   │   ├── upsample_layer.cpp
    │   │   ├── upsample_layer.h
    │   ├── Makefile
    │   ├── nvdsinfer_yolo_engine.cpp
    │   ├── nvdsinitinputlayers_Yolo.cpp
    │   ├── nvdsparsebbox_Yolo.cpp
    │   ├── nvdsparsebbox_Yolo_cuda.cu
    │   ├── utils.cpp
    │   ├── utils.h
    │   ├── yolo.cpp
    │   ├── yoloForward.cu
    │   ├── yoloForward_nc.cu
    │   ├── yoloForward_v2.cu
    │   ├── yolo.h
    │   ├── yoloPlugins.cpp
    │   ├── yoloPlugins.h
    └── scripts
        └── export_yolo11.py
```

<br>

## 🚀  Quickstart
```bash
# clone project
git clone https://github.com/sh-aidev/deepstream-yolo11.git
cd deepstream-yolov11

# Download the model weights
cd src
wget wget -P models/ https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt

# For Other yolo weights use the below links:
# YOLO11n - https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
# YOLO11s - https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt
# YOLO11l - https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt
#YOLO11x - https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt

# Convert the model weights to onnx
python3 scripts/export_yolo11.py -w models/yolo11m.pt

# Change the model path in the config_infer_primary_yoloV11.txt file to the onnx file generated as
# onnx-file=models/yolo11m.onnx

# Set the CUDA_VER according to your DeepStream version
export CUDA_VER=12.2

# COmpile the YOLO plugin
make -C nvdsinfer_custom_impl_Yolo clean && make -C nvdsinfer_custom_impl_Yolo

# Run Deepstream Application
deepstream-app -c deepstream_app_config.txt
```

<br>

## 📝  Docker container usage instructions
**Prerequisites:**
- [Docker](https://docs.docker.com/get-docker/)
- [Visual Studio Code](https://code.visualstudio.com/)
- [Remote - Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension

**Steps:**
1. Clone this repository
2. Open the repository in Visual Studio Code
3. press crtl+shift+p and select "Remote-Containers: Reopen in Container"
4. Wait for the container to build
5. Open a terminal in Visual Studio Code and run the following commands:

<br>

## References
- [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo.git)
- [DeepStream SDK Documentation](https://docs.nvidia.com/metropolis/deepstream/dev-guide/index.html)
- [NVIDIA DeepStream GitHub](https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps)
- [YOLOv11 Ultralytics](https://docs.ultralytics.com/models/yolo11/)
<br>