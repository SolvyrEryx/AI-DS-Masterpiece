<div align="center">

<!-- Header with Deep Space Purple Theme -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,18,24&height=280&section=header&text=AI%20%26%20Data%20Science&fontSize=90&animation=fadeIn&fontAlignY=38&desc=Masterpiece%20Collection&descAlignY=60&descAlign=50" width="100%"/>

<!-- Animated Typing Title -->
<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=24&duration=3000&color=DE29FA&center=true&vCenter=true&lines=Advanced+AI+%26+Deep+Learning+🔥;CNN+Image+Classifier+%7C+PyTorch+💫;High-Performance+Neural+Networks" alt="Typing SVG" />

<!-- Cosmic Divider -->
<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="900">

</div>

---

## 🌌 About This Repository

<img align="right" alt="AI Coding" width="400" src="https://user-images.githubusercontent.com/74038190/229223263-cf2e4b07-2615-4f87-9c38-e37600f8381a.gif">

### 🎯 Mission Statement

Diving deep into the **infinite universe of AI and Data Science**, this repository showcases cutting-edge projects that blend:

- 🤖 **Machine Learning** algorithms  
- 📊 **Advanced Data Analytics**  
- 🧪 **Deep Learning Models**  
- 🎨 **Creative Data Visualization**  
- ⚡ **High-Performance Computing**

<br clear="right"/>

---

<div align="center">

## 💫 Featured: MNIST CNN Classifier

<img src="https://user-images.githubusercontent.com/74038190/212257454-16e3712e-945a-4ca2-b238-408ad0bf87e6.gif" width="100">
<img src="https://user-images.githubusercontent.com/74038190/212257472-08e52665-c503-4bd9-aa20-f5a4dae769b5.gif" width="100">
<img src="https://user-images.githubusercontent.com/74038190/212257468-1e9a91f1-b626-4baa-b15d-5c385dfa7ed2.gif" width="100">
<img src="https://user-images.githubusercontent.com/74038190/212257465-7ce8d493-cac5-494e-982a-5a9deb852c4b.gif" width="100">

</div>

### 🐍 Python Code: `mnist_classifier.py`

**Sophisticated Convolutional Neural Network for handwritten digit recognition**

#### ✨ Features:
- 🎯 **Advanced CNN Architecture** with 3 convolutional layers
- 🔥 **Batch Normalization** for improved training stability
- 💎 **Dropout Regularization** to prevent overfitting
- 📊 **Real-time Training Visualization** with progress bars
- 🎨 **Beautiful Plotting** of training history and predictions
- ⚡ **GPU Support** for accelerated training
- 🏆 **99%+ Accuracy** on MNIST test set

#### 🚀 Quick Start:

```bash
# Clone the repository
git clone https://github.com/SolvyrEryx/AI-DS-Masterpiece.git
cd AI-DS-Masterpiece

# Install dependencies
pip install torch torchvision matplotlib numpy tqdm

# Run the classifier
python mnist_classifier.py
```

#### 📦 Model Architecture:

```python
EnhancedCNN(
  (conv1): Conv2d(1, 32, kernel_size=3, padding=1)
  (conv2): Conv2d(32, 64, kernel_size=3, padding=1)
  (conv3): Conv2d(64, 128, kernel_size=3, padding=1)
  (bn1): BatchNorm2d(32)
  (bn2): BatchNorm2d(64)
  (bn3): BatchNorm2d(128)
  (pool): MaxPool2d(2, 2)
  (fc1): Linear(1152, 256)
  (fc2): Linear(256, 128)
  (fc3): Linear(128, 10)
  (dropout): Dropout(p=0.5)
)
```

#### 📈 Expected Output:

```
======================================================================
               MNIST CNN CLASSIFIER
          AI-DS-Masterpiece by Solvyr Eryx
======================================================================

Hyperparameters:
  Batch Size: 64
  Learning Rate: 0.001
  Epochs: 10
  Device: cuda

Training samples: 60000
Test samples: 10000

Total parameters: 1,199,882

Epoch [10/10]
----------------------------------------------------------------------
Training: 100%|██████████| 938/938 [00:45<00:00, 20.84it/s]
Evaluating: 100%|██████████| 157/157 [00:03<00:00, 44.21it/s]

Train Loss: 0.0234 | Train Acc: 99.28%
Test Loss: 0.0289 | Test Acc: 99.12%

======================================================================
Training completed in 8.45 minutes
Final Test Accuracy: 99.12%
======================================================================
```

#### 🎨 Output Visualizations:

The script generates two stunning visualizations:
1. **training_history.png** - Loss and accuracy curves with deep space purple theme
2. **predictions.png** - Sample predictions with color-coded results

---

<div align="center">

## 💻 Tech Stack

<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" alt="Python" width="60" height="60"/>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pytorch/pytorch-original.svg" alt="PyTorch" width="60" height="60"/>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/tensorflow/tensorflow-original.svg" alt="TensorFlow" width="60" height="60"/>
  <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" alt="Sklearn" width="60" height="60"/>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/opencv/opencv-original.svg" alt="OpenCV" width="60" height="60"/>
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/docker/docker-original.svg" alt="Docker" width="60" height="60"/>
</p>

<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
<img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
<img src="https://img.shields.io/badge/Deep_Learning-7209B7?style=for-the-badge&logo=deeplearning&logoColor=white" />
<img src="https://img.shields.io/badge/Computer_Vision-9D4EDD?style=for-the-badge&logo=opencv&logoColor=white" />

</div>

---

<div align="center">

## 🌟 Project Structure

```
AI-DS-Masterpiece/
│
├── mnist_classifier.py    # Main CNN classifier script
├── README.md             # This file
├── data/                 # MNIST dataset (auto-downloaded)
├── mnist_cnn_model.pth   # Trained model weights
├── training_history.png  # Training visualization
└── predictions.png       # Sample predictions
```

</div>

---

<div align="center">

## 🚀 Coming Soon

- 🔮 Advanced NLP Models with Transformers
- 🎯 Object Detection with YOLO
- 📊 Time Series Forecasting
- 🎨 GANs for Image Generation
- 🧠 Reinforcement Learning Agents

---

<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="900">

### 💜 Built with passion by [Solvyr Eryx](https://github.com/SolvyrEryx)

<img src="https://komarev.com/ghpvc/?username=SolvyrEryx&label=Repository%20Views&color=9d4edd&style=for-the-badge" alt="Views" />

<!-- Animated Footer -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,18,24&height=120&section=footer&animation=twinkling" width="100%"/>

</div>
