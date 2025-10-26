# Sign-Language-Detection


## 📌 Overview
Sign Language Detection is a real-time system that recognizes hand gestures and converts them into readable text using computer vision and machine learning. This project aims to bridge the communication gap between the hearing/speech-impaired community and non-signers by using webcam-based gesture recognition without requiring any external sensors or gloves.

## 🎯 Objective
- Detect and classify sign language gestures in real-time.  
- Build an AI-based communication tool using computer vision and deep learning.  
- Provide an accessible solution for inclusive communication.

## 🛠️ Technologies Used

| Category               | Tools / Libraries          |
|------------------------|-----------------------------|
| Programming Language   | Python                     |
| Image Processing       | OpenCV, MediaPipe, CVZone  |
| Machine Learning       | TensorFlow / Keras         |
| Data Handling          | NumPy, Pandas              |
| Visualization          | Matplotlib                 |
| Model Format           | `.h5` (Keras saved model)  |

## 📂 Project Structure

```
Sign-Language-Detection/
│
├── datacollection.py         # Script to collect gesture images
├── test.py                   # Real-time sign language prediction
├── model.h5                  # Trained deep learning model
├── /dataset/                 # Stored gesture images (class-wise)
└── README.md                 # Project documentation
```

## ⚙️ How the System Works

### 1️⃣ Data Collection
- Images of hand gestures are captured using a webcam.  
- Each gesture (like A-Z, Hello, Yes) is saved in a separate folder.

### 2️⃣ Preprocessing
- Resize and normalize images.  
- Extract hand landmarks using MediaPipe or CVZone.

### 3️⃣ Model Training
- A Convolutional Neural Network (CNN) is trained on gesture images.  
- The model learns to map gestures to corresponding labels.

### 4️⃣ Real-Time Detection
- Live video is captured using a webcam.  
- Frame is sent to the trained model for prediction.  
- Output is displayed as text on screen.

## 🚀 How to Run the Project

### ✅ 1. Clone the Repository
```bash
git clone https://github.com/joshitha5g4/Sign-Language-Detection.git
cd Sign-Language-Detection
```

### ✅ 2. Install Dependencies
```bash
pip install opencv-python mediapipe cvzone tensorflow numpy
```

### ✅ 3. Collect Gesture Data (Optional)
```bash
python datacollection.py
```

### ✅ 4. Run Real-Time Sign Detection
```bash
python test.py
```

## 🌟 Features
✔ Real-time gesture detection using webcam  
✔ No gloves or sensors required  
✔ High accuracy with CNN and landmark detection  
✔ Easy to add new gestures  
✔ Beginner-friendly and modular code

## 📉 Limitations
- Background noise and lighting variations may affect accuracy  
- Limited number of gestures unless dataset is expanded  
- Requires proper camera positioning and clear hand visibility  

## 🔮 Future Enhancements
🔹 Add voice output for detected gestures  
🔹 Recognize dynamic gestures (continuous motion)  
🔹 Support full sentences and Indian Sign Language (ISL)  
🔹 Deploy as a web or mobile application  

✅ Conclusion
This project successfully demonstrates a real-time Sign Language Detection system using deep learning and computer vision. By combining technologies like OpenCV, CVZone, and TensorFlow, it enables gesture recognition through a simple webcam—making communication more accessible for individuals with hearing or speech impairments. Although the system currently supports a limited set of gestures, it provides a strong foundation for developing more advanced and inclusive solutions. With further improvements such as voice integration, additional gestures, and deployment on web or mobile platforms, this project has the potential to create a meaningful impact in real-world accessibility and assistive technology.

⭐ If you like this project, don’t forget to star the repository!
