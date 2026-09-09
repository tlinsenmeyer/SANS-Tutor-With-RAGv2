# SANS-Tutor-With-RAGv2

A lightweight, local AI tutoring system focused on cybersecurity concepts, built using SGLang and a Retrieval-Augmented Generation (RAG) pipeline. The project runs fully inside Docker + WSL2 with GPU acceleration and provides a clean start/stop workflow on Windows.

---

## 🚀 Overview

SANS-Tutor-With-RAGv2 is a local, containerized AI assistant designed to help with cybersecurity learning and practice.  
It uses:

- **SGLang** for fast GPU inference  
- **RAG** for retrieving relevant cybersecurity content  
- **WSL2 + Docker** for a clean, reproducible runtime  
- **Windows launcher scripts** for simple start/stop behavior  

---

## 📦 Components

- **app/** — Main application logic (agents, RAG pipeline, API)
- **sglang/** — SGLang source + Dockerfile for GPU inference
- **compose.yaml** — Docker Compose configuration
- **service/sglang.service** — Optional systemd auto-start inside WSL
- **scripts/** — Windows `start.bat` and `stop.bat` launchers

---

## 🔧 Requirements

- Windows 11  
- WSL2 with Ubuntu 24.04  
- Docker Desktop (WSL2 backend)  
- NVIDIA GPU + WSL CUDA drivers  

---

## 🛠️ Build & Run

### Build the SGLang image:

```bash
docker buildx build --load \
  --memory=32g \
  --shm-size=16g \
  -t sglang-server \
  -f sglang/docker/Dockerfile .




