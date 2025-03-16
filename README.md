# NeuroLens: Image Captioning with CNN + LSTM Attention and Decoder Architecture

NeuroLens is an image captioning project that leverages deep learning techniques—combining Convolutional Neural Networks (CNN) for image feature extraction with Long Short-Term Memory (LSTM) networks and an attention mechanism for natural language generation. The model is deployed as a RESTful API using Flask, containerized with Docker, and hosted on AWS ECS Fargate.


## Table of Contents
- [Project Overview](#project-overview)
- [Model Architecture](#model-architecture)
  - [CNN Feature Extraction](#cnn-feature-extraction)
  - [LSTM Decoder with Attention](#lstm-decoder-with-attention)
- [Model Training](#model-training)
- [Inference Pipeline](#inference-pipeline)
- [Deployment](#deployment)
  - [Docker Container](#docker-container)
  - [AWS ECS Fargate](#aws-ecs-fargate)
- [API Usage](#api-usage)
- [Requirements](#requirements)
- [Setup](#setup)
  - [Installation](#installation)


## Project Overview

NeuroLens generates descriptive captions for images by:

  - Extracting image features using a pre-trained CNN (InceptionV3).

  - Feeding those features into an LSTM-based decoder equipped with an attention mechanism.

  - Producing a natural language caption that describes the image content.

The project is deployed as a Flask application, packaged in a Docker container, and orchestrated on AWS ECS Fargate for scalable, serverless inference.


## Model Architecture

### CNN Feature Extraction

- **InceptionV3**:

  Pre-trained on ImageNet, the InceptionV3 CNN is used to extract image features from the last convolutional layer before the fully connected layers.

  - **How it Works**:

    - Images are resized to 299×299.

    - They are preprocessed (normalized) using InceptionV3’s `preprocess_input` function.

    - Features are extracted from the penultimate layer of the network, yielding a 2048-dimensional vector for each image.

### LSTM Decoder with Attention

- **Decoder**:

  The extracted CNN features are passed to an LSTM decoder that uses an attention mechanism. This decoder generates the image caption word by word.

- **Attention Mechanism (Bahdanau Attention)**:

  - **Purpose**:

    Focuses on different parts of the image feature vector during the generation of each word.

  - **How it Works**:
    - The attention mechanism computes attention weights for each part of the image feature vector.

    - These weights are used to compute a context vector, which is concatenated with the word embedding of the previous word to guide the Decoder and predict the next word in the caption.

- **LSTM-based Decoder**:

  - Receives the context vector and previous word (initially a start token).

  - Uses an LSTM (or GRU) layer to predict the next word.

  - Continues generating words until an end token is produced or a maximum caption length is reached.


## Model Training

**During training**:

  - The image captioning model is built using a CNN encoder (pre-trained InceptionV3) and an LSTM decoder with attention

  - Captions are tokenized and padded.

  - Teacher forcing is used during training to feed ground-truth tokens to the decoder.

  - The model is optimized using a suitable loss function (sparse categorical cross-entropy).

  - After training, the model weights are saved in an H5 file, and the tokenizer is saved as a pickle file.


## Inference Pipeline

**During inference**:

  1. **Image Preprocessing**:

     - Images are resized to 299×299.

     - They are preprocessed using InceptionV3’s preprocess_input function.

  2. **Feature Extraction**:

     - The preprocessed image is passed through the InceptionV3 CNN to extract image features.

  3. **Caption Generation (Greedy Decoding)**:

     - The decoder starts with a start token.

     - The attention mechanism guides the decoder to focus on relevant image regions.

     - The decoder generates one word at a time until it produces an end token or reaches the maximum length.

  4. **Output**:

     - The generated caption (without start/end tokens) is returned as a JSON response.


## Deployment

### Docker Container

The project is containerized using Docker. The Dockerfile:

  - Uses an official Python slim image.

  - Installs required packages (Flask, TensorFlow, Pillow, NumPy, Gunicorn, etc.).

  - Copies the `models` folder (with the saved H5 and tokenizer files) and the src folder (containing the inference Flask code) into the container.

  - Uses Gunicorn to serve the Flask app on port 5000.

### AWS ECS Fargate

  - **Task Definition**:

    An ECS task definition is configured to use Fargate, specifying resource requirements, port mappings, and the container image (pushed to Amazon ECR).

  - **Deployment**:

    The Docker container is deployed on AWS ECS Fargate, making the API available at `http://35.175.219.65:5000/predict`

  - **Health Check**:

    A `/health` endpoint is provided for ECS to monitor the service’s status.


## API Usage

`/predict` Endpoint

  - Method: POST

  - URL: `http://35.175.219.65:5000/predict`

  - Body: Multipart form-data with a key `"image"` containing the image file.

  - Response Example:

    ```json
    {
      "caption": "a dog playing in the grass"
    }
    ```

`health` Endpoint

  - Method: GET

  - URL: `http://35.175.219.65:5000/health`

  - Response:

      ```json
      {
        "status": "ok"
      }
      ```


## Requirements

 - **Flask**: Provides the web API.

 - **TensorFlow**: Used for model building, training, and inference (including InceptionV3).

 - **Pillow**: For image processing.

 - **NumPy**: For numerical operations.

 - **Gunicorn**: WSGI server for production deployment.

 - **Werkzeug**: (Specified version compatible with Flask)


## Setup

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Eldrago12/NeuroLens.git
   cd NeuroLens
   ```

2. **Build the Docker Image**:

   ```bash
   docker build -t neuro-lens .
   ```

3. **Run the Docker Container**:

  ```bash
  docker run -d -p 5000:5000 --name neurolens-container neuro-lens
  ```

4. **Verify the Container is Running**:

  ```bash
  docker ps
  ```

5. **Make a Prediction Using an Image**:

  Use `curl` to send an image file to the API:

  ```bash
  curl -X POST -F "image=@path/image.jpg" http://localhost:5000/predict
  ```

  Expected response (example):

  ```json
  {
    "caption": "a cat sitting on a sofa"
  }
  ```

6. **Stopping and Removing the Container**:

   To stop the running container:

   ```bash
   docker stop neurolens-container
   ```

   To remove the container:

   ```bash
   docker rm neurolens-container
   ```

7. **Removing the Docker Image (If Needed)**:

   ```bash
   docker rmi neurolens-app
   ```

## Contribute & Connect

If you find any improvements or have better approaches, feel free to contribute! 🚀

Let's connect and discuss further optimization of this project:

- **LinkedIn**: [Sirshak Dolai](https://www.linkedin.com/in/sirshak-dolai)


⭐️ **If you find this repo helpful, consider giving it a star!** ⭐️
