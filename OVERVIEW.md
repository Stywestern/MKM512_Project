# Sentry Turret

![Sentry Turret Physical Chassis](assets/media/system_photos/chassis_front.jpg)
[![Watch the Sentry Turret Demo Video](assets/media/system_photos/video_thumbnail.jpg)](https://www.youtube.com/watch?v=your_video_id)

## Abstract & Overview
This repository contains the software framework for an autonomous Sentry Turret developed at Istanbul Technical University by Alper Aktören (mechanical), Cem Baltacı (electrical and integration), Eren Açıkgöz (control), and Kerem Cantimur (software), under the guidance of Prof. Dr. Volkan Sezer and Araş. Gör. Ahmet Kağızman. 

The system utilizes a multi-threaded computer vision pipeline to detect, track, and recognize faces defined in a local dataset. While targeting and positioning are fully autonomous, engagement execution (such as directing a laser pointer or firing mechanism) relies on a human operator to maintain a safe, human-in-the-loop configuration. The physical sentry was designed with a custom 3D-printed chassis, using an industrial PLC module to drive the high-torque pan/tilt motor assembly.

For detailed methodological explanations, kinematics, and performance analysis, please refer to our paper: `[paper placeholder]`.

---

## Prerequisites

### Hardware Requirements
* **PLC / Controller:** The original project utilized an **Omron NX1P2 PLC** module for robust motor execution. While a simpler microcontroller can be adapted, it introduces challenges regarding real-time task scheduling that must be accounted for.
* **Actuators:** High-torque **24V Servo Motors** powering the main pan and tilt gear assemblies.
* **Camera:** **A4Tech PK-910H USB Webcam**. The vision pipeline requires a camera capable of delivering a stable stream of at least 24 FPS at 480p resolution, though the underlying neural networks are robust across varied distances and lighting conditions.
* **Laser/Relay:** A 5V low-power point laser diode triggered via an isolated intermediate hardware relay module.

### Software & Environment Requirements
* **Python Version:** `3.12.9`
* **Machine Learning:** `PyTorch` (CUDA-accelerated version highly recommended for real-time inference framework execution).
* **Computer Vision:** `opencv-python` for real-time frame capture, manipulation, and coordinate alignment.
* **User Interface:** `PyQt6` for rendering the asynchronous operator dashboard, HUD graphics, and hardware toggles.

*Note: The software architecture is highly modular; dependencies and hardware abstraction layers can be adapted or swapped with minimal friction.*

### Networking & I/O
* An active **Ethernet connection** configured to communicate with the PLC over TCP/IP.
* A dedicated **USB 3.0 / 2.0 port** for uncompressed webcam video data transmission.

---

## Quickstart Guide

### 1. Installation
Clone the repository and install the verified library dependencies using your preferred environment manager:
```bash
git clone <repository-url>
cd <project-folder>
pip install -r requirements.txt
```

### 2. Dataset Preparation
To be able to run tracking and detections, you will need to create an embedding dataset. For this, navigate to assets/faces/raw_images and populate that folder with pictures of the people that you want to be recognized. Simply create a folder with your preferred name for the target (e.g. Adem_Ademoğlu) and put between 10-15 pictures of said target -preferably under different lightings and from different angles-. After populating this folder, run: 

```bash
python src/face_embeddings.py
```

You should expect two more folders created under assets/, debug_aligned which holds the aligned versions of the pictures and embeddings which holds the pickle file that stores said embeddings with their labels -same label as the folder name-.

### 3. Running
Run via:
```bash
python src/main_gui.py
```

You should expect a UI window to open with camera feed on the upper-left and working buttons. You may refer to the figures and the the video.

# Project Folder Structure
## `assets/`
Contains datasets and models.

* **`faces/`**
    * **`raw_images/`**: Contains the images of people represented in the dataset. Each directory contains 10-15 pictures of said person.
    * **`debug_aligned/`**: Contains the aligned images (using SCRFD landmark alignment) of people represented in the dataset.
    * **`embeddings/`**: This contains the embedding vectors for all images in the dataset. These vectors are what ArcFace has produced, a 512-dimensional feature vector embedding of a singular aligned image. The recordings are labeled with the name of the `.jpg` file.
* **`models/`**
    * Contains the Deep Neural Network models used in the pipeline (SCRFD, ArcFace, optional RE-ID model for BoT-SORT).

---

## `src/`
Source code. The project utilizes a modular architecture, employing a central Coordinator/Mediator pattern (main_gui.py) to separate software logic from the hardware control flow.

* `main_gui.py`: Orchestrator file. Its main job is to initialize and set the communications between the modules that perform the utilities.
* `config.py`: Holds settings to change throughout the project, such as the camera aspect ratios or who is marked as enemy or friend.
* `face_embeddings.py`: Used to create the embeddings for the raw images before starting the software. The created embeddings are stored in a pickle file under `assets/embeddings`.

### `src/modules/`
Contains the modules that try to perform singular and modular utilities.

* `utils.py`: Contains common utility functions, such as an overloaded print function.
* `camera.py`: Contains the `CameraStream` class that works on its own Camera thread. Its main purpose is to communicate with the external camera through the USB connection, pull the frames in its maximum FPS capability, and give them to the VisionWorker thread.
* `visionworker.py`: Contains the bulk of the project's code and runs on its own Worker thread. Worker utilizes other modules to run the detection -> tracking -> recognition pipeline. It is also responsible for the functionality of the interface buttons.
* `interface.py`: The objective of this module is to construct the User Interface (UI), which includes setting up the boxes and the buttons. Works on the main App thread. The buttons perform duties defined in the VisionWorker. Frames from the `CameraStream` are also shown in this app at the upper-left box.
* `detector.py`: This module's objective is to process the current frame, detect faces, and pass that information. For every face it detects in a given frame, it will output the box that is bounding it and 5 facial landmarks (the eyes, the nose, the tips of the mouth). In the original project, this module uses the SCRFD neural network model.
* `tracker.py`: This module takes the bounding box as an input and produces unique IDs for targets detected, and continues this process until the target is out of our detection sights. The output will contain which box was which ID between the frames. In the original project, this module uses the BoT-SORT tracking algorithm, with the Re-ID feature turned off.
* `recognizer.py`: This module uses the bounding box and facial landmarks that the detector module creates and after aligning the image to fit its architecture, it creates an embedding. VisionWorker then uses this embedding to run similarity checks on the dataset.
* `PLC.py`: The original project used a PLC module to power and control the motors of the sentry. This class is meant to construct the communication protocols between the mother computer that runs the UI and the AI models, and the aforementioned PLC module.
* `controller.py`: Contains a simple PD controller class that helps the physical tracking of locked targets when given a command. The controller works on its own async Controller thread so that the motors don't jitter too much because of a delay. It pulls the last command it received, sometimes skipping the commands of some frames if there was a computational overload.

---

## `testing/`
This folder contains singular tests about the modules to better understand their standalone capabilities and to alleviate bugs.




