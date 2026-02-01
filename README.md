# Face-Recognition-System
Real-time face detection

## Modes

### 1. Face Registration
- Captures face images from the camera
- Saves images per user ID
- Press `c` to capture a face
- Press `q` to exit registration

### 2. Model Training
- Loads stored face images
- Trains LBPH face recognizer
- Saves trained model to disk

### 3. Face Recognition
- Detects and recognizes faces in real time
- Displays user ID and confidence score
- Lower confidence indicates better match

---

## Prerequisites

- Linux (Ubuntu 20.04+ recommended)
- USB camera
- g++ compiler
- OpenCV 4.x

---

## Install OpenCV

sudo apt update
sudo apt install libopencv-dev

Verify installation:
pkg-config --modversion opencv4

Haar Cascade Setup
cp /usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml .





