## 🌿 Plant Disease Identification using Leaf Images

### 📋 Overview

This project implements a **Convolutional Neural Network (CNN)** using Keras and TensorFlow to classify plant leaf diseases. The primary goal is to provide a fast and accurate prediction system through an interactive web application built with **Streamlit**.

The project consists of two main parts:

1.  **Model Training (`Plant_Disease_Detection.py`):** The script used to prepare the dataset, define the CNN architecture, and train the model.
2.  **Web Application (`main_app.py`):** A Streamlit application that allows users to upload a leaf image and get an instant disease prediction.

-----

### 🚀 Getting Started

#### Prerequisites

To run this project locally, you need Python and the following libraries. It is recommended to use a virtual environment.

```bash
# Example: Creating and activating a virtual environment
python -m venv venv
source venv/bin/activate 
```

#### Installation

Install the required packages using pip:

```bash
pip install numpy streamlit opencv-python tensorflow keras matplotlib scikit-learn pandas
```

#### Running the Application

To start the interactive web interface, navigate to the project's root directory in your terminal and run:

```bash
streamlit run main_app.py
```

This command will open the application in your web browser, allowing you to upload images for prediction.

-----

### 🧠 Model Training Details

The model was trained using the `Plant_Disease_Detection.py` script.

#### Architecture

The model uses a sequential **Convolutional Neural Network (CNN)** architecture:

| Layer Type | Parameters | Output Shape | Notes |
| :--- | :--- | :--- | :--- |
| **Conv2D** | 32 filters, (3, 3) kernel | (128, 128, 32) | Input shape is fixed at (128, 128, 3). Uses `relu`. |
| **MaxPooling2D**| (3, 3) pool size | | Reduces dimensionality. |
| **Conv2D** | 16 filters, (3, 3) kernel | | Uses `relu`. |
| **MaxPooling2D**| (2, 2) pool size | | Further reduces dimensionality. |
| **Flatten** | - | | Prepares data for Dense layers. |
| **Dense** | 8 units | | Uses `relu`. |
| **Dense** | 15 units | | **Output layer.** Uses `softmax` for 15 classes. |

#### Configuration

  * **Input Image Size:** 128x128 pixels (3 color channels).
  * **Optimizer:** Adam with a learning rate of `0.0001`.
  * **Loss Function:** `categorical_crossentropy`.
  * **Epochs:** 50
  * **Batch Size:** 128

-----

### 🖼️ Supported Classes

The model is trained to identify 15 distinct classes across Corn (Maize), Grape, Soybean, and Tomato plants.

| Plant | Disease/Status |
| :--- | :--- |
| **Corn (Maize)** | Cercospora leaf spot Gray leaf spot |
| **Corn (Maize)** | Common rust |
| **Corn (Maize)** | healthy |
| **Corn (Maize)** | Northern Leaf Blight |
| **Grape** | Black rot |
| **Grape** | Esca (Black Measles) |
| **Grape** | healthy |
| **Grape** | Leaf blight (Isariopsis Leaf Spot) |
| **Soybean** | Septoria Brown Spot |
| **Soybean** | Vein Necrosis |
| **Soybean** | healthy |
| **Tomato** | Bacterial spot |
| **Tomato** | Early blight |
| **Tomato** | healthy |
| **Tomato** | Late blight |
