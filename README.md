# AI Vision & Sign Language Translator 🤟

A modern, web-based interface that leverages machine learning to classify images in real-time and provide text-to-sign language translation utilities. Built with **TensorFlow.js** and **Teachable Machine**, this project runs entirely in the browser for high privacy and low latency.

## 🚀 Features

* **Real-time Classification:** Uses a custom-trained Teachable Machine model to identify signs/objects via webcam.
* **Visual Feedback:** High-precision probability bars with automatic highlighting of the most likely result (threshold > 80%).
* **Text-to-Sign Module:** A dedicated interface to translate written text into sign language representations.
* **Responsive Dark UI:** A sleek, glassmorphism-inspired dashboard optimized for both desktop and mobile viewing.
* **Privacy First:** All AI processing happens locally on your device; no video data is sent to a server.

## 🛠️ Tech Stack

* **Frontend:** HTML5, CSS3 (Custom Variables & Flexbox)
* **Machine Learning:** [TensorFlow.js](https://www.tensorflow.org/js)
* **Model Source:** [Teachable Machine by Google](https://teachablemachine.withgoogle.com/)

## 📂 Project Structure

```text
├── index.html              # Main AI Vision Dashboard
├── text-to-sign.html       # Text-to-Sign Translation Page
├── my_model/               # Local Model directory
│   ├── model.json          # Model topology
│   ├── metadata.json       # Class labels
│   └── weights.bin         # Learned parameters
└── README.md               # Project documentation
