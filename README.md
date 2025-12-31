# 🚀 AI-Powered Defect Detection & Quote Generation System

**A dual-purpose AI application combining computer vision and natural language generation**

---

## 🎥 Project Demonstration

<p align="center">
  <a href="https://www.youtube.com/watch?v=W2tcrdUNaYM">
    <img src="https://img.youtube.com/vi/W2tcrdUNaYM/maxresdefault.jpg" width="600" />
  </a>
</p>

*Click the image above to watch the full demonstration on YouTube*

---

## 🌟 Overview

This project showcases two distinct AI capabilities within a single FastAPI application:

1. **🔍 Industrial Defect Detection System** - Real-time surface defect identification using YOLO
2. **✨ Custom Language Model** - GPT-2 inspired architecture for motivational quote generation

Built from the ground up to demonstrate end-to-end ML engineering skills, from model training to production deployment.

---

## 🎯 Features

### Vision Module: YOLO Defect Detector

Detects and localizes surface defects in industrial materials across **6 distinct categories**:

- 🕸️ **Crazing** - Fine surface cracks
- 🔘 **Inclusion** - Foreign material embedded in surface
- 🕳️ **Pitted Surface** - Small cavities or depressions
- 🩹 **Patches** - Irregular surface patches
- 📊 **Rolled-in Scale** - Scale pressed into surface during rolling
- 💥 **Scratches** - Linear surface damage

**Key Capabilities:**
- Real-time bounding box visualization
- Multi-class detection in single image
- Confidence score reporting
- RESTful API integration

### Language Module: Custom LLM Quote Generator

A language model built from scratch using GPT-2 architecture principles:

**Training Pipeline:**
- **Base Training:** TinyShakespeare dataset for foundational language understanding
- **Fine-tuning:** LoRA (Low-Rank Adaptation) on curated quote dataset
- **Specialization:** Optimized for motivational content generation

**Supported Keywords:**
- 💪 Motivation
- 🦁 Courage  
- 🎯 Focus
- 🌱 Resilience
- 🚀 Ambition
- *...and more*

> **Note:** As a lightweight model trained from scratch, outputs may occasionally be incoherent or hallucinate. Best results achieved with trained keywords.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   FastAPI Backend                    │
├──────────────────────┬──────────────────────────────┤
│   YOLO Detector      │   Custom LLM (GPT-2 Style)   │
│   ├─ Image Input     │   ├─ Keyword Input           │
│   ├─ Preprocessing   │   ├─ Token Generation        │
│   ├─ Inference       │   ├─ LoRA Weights            │
│   └─ Bbox Output     │   └─ Text Output             │
└──────────────────────┴──────────────────────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │  Docker Container│
              └─────────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │  Render Platform │
              └─────────────────┘
```

---

## 🛠️ Tech Stack

**Core Framework:**
- FastAPI for high-performance API endpoints
- Uvicorn ASGI server

**AI/ML:**
- PyTorch for model development
- YOLO for object detection
- Custom Transformer implementation (GPT-2 architecture)
- LoRA for efficient fine-tuning

**Deployment:**
- Docker for containerization
- Render for cloud hosting
- GitHub Actions for CI/CD (optional)

---

## 🚀 Quick Start

### 🌐 Live Application

The application is deployed and ready to use - no installation required!

Simply visit the links:
- Test the defect detection on your images
- Generate motivational quotes with custom keywords
- Explore the interactive Swagger UI

---

## 🧠 Model Details

### YOLO Defect Detector
- **Architecture:** YOLOv8
- **Training Dataset:** NEU Surface Defect Database
<!-- - **Performance:** mAP@0.5: XX% -->
<!-- - **Inference Time:** ~XX ms per image -->

### Custom Language Model
- **Architecture:** GPT-2 inspired Transformer
- **Vocab Size:** 50257
- **Context Length:** 128 tokens
- **Training:**
  - Base: TinyShakespeare (1M characters)
  - Fine-tuning: Quote dataset (1007 examples)
- **Fine-tuning Method:** LoRA (rank=8, alpha=16)

---

## 🎓 Learning Outcomes

This project demonstrates proficiency in:

✅ **Deep Learning:** Implementing transformer architectures from scratch  
✅ **Computer Vision:** YOLO-based object detection  
✅ **NLP:** Language model training and fine-tuning  
✅ **MLOps:** Model deployment and containerization  
✅ **API Development:** RESTful services with FastAPI  
✅ **DevOps:** Docker, cloud deployment (Render)  
✅ **Transfer Learning:** LoRA fine-tuning techniques  

---

## 📝 Known Limitations

- **LLM Output Quality:** Being a small model trained from scratch, the quote generator may produce incoherent or repetitive text, especially with keywords outside the training distribution
- **Inference Speed:** Current deployment on free tier may have cold start delays
- **Model Size:** Limited context window (20-30 tokens) for LLM

---

## 🙏 Acknowledgments

- TinyShakespeare dataset by Andrej Karpathy
- NEU Surface Defect Database
- Ultralytics YOLO
- Hugging Face PEFT library for LoRA implementation

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ and ☕

</div>