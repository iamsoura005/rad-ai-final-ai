# 🩺 RadiologyAI

An advanced, full-stack medical analysis and triage platform. RadiologyAI combines a responsive React frontend with a FastAPI backend powered by Groq's high-speed Llama-3/Llama-4 models to provide structured observations and clinician-focused insights.

---

## ✨ Features

- 🔬 **Scan Analysis:** Upload medical scans (X-rays, MRI, CT scans, ultrasounds) for automated, structured radiology observations, severity assessments, and clinical explanations.
- 🩺 **Symptom Checker:** Interactive triage tool designed to parse symptoms, estimate urgency levels, list potential causes, and outline clear next steps.
- 💊 **Medicine Info:** Search tool to quickly query drug classes, uses, dosages, side effects, contraindications, and warnings.
- 💬 **Health Chat:** A compassionate, context-aware clinical assistant (Dr. AI) to answer general medical queries.
- 💾 **Patient History:** Locally stored session tracking to review past scans, chat histories, and triage reports.

---

## 🛠️ Technology Stack

| Component | Technologies Used |
| :--- | :--- |
| **Frontend** | React, Vite, TailwindCSS, Framer Motion, Lucide Icons |
| **Backend** | FastAPI, Python, Uvicorn, Python-Dotenv, Groq SDK |
| **Hosting** | Vercel Serverless (Single-project frontend & API routing) |

---

## 🚀 Quick Start (Local Development)

### Prerequisites
- Node.js (v18+)
- Python (3.10+)

### Setup Steps
1. **Clone the Repository**
   ```bash
   git clone https://github.com/iamsoura005/rad-ai-final-ai.git
   cd rad-ai-final-ai
   ```

2. **Configure local environment variables**
   Create a `.env` file inside the `backend/` directory:
   ```bash
   # Add your Groq API Key to backend/.env
   GROQ_API_KEY=your_groq_api_key_here
   ```

3. **Run using Startup Script** (Windows PowerShell)
   ```powershell
   .\start-dev.ps1
   ```
   *This script automatically copies the env placeholder, installs python requirements, installs npm packages, and starts both backend (port 8000) and frontend (port 5173).*

4. **Run Manually**
   - **Backend:**
     ```bash
     cd backend
     pip install -r requirements.txt
     uvicorn main:app --reload --port 8000
     ```
   - **Frontend:**
     ```bash
     cd frontend
     npm install
     npm run dev
     ```

---

## 📂 Project Architecture

```
rad-ai-final-ai/
├── api/                  # Vercel serverless function entry points
│   ├── index.py          # Routes /api requests to FastAPI
│   └── requirements.txt  # Serverless dependencies (optimized)
├── backend/              # FastAPI application
│   ├── main.py           # Core backend logic & Groq integration
│   ├── region_parser.py  # Text parser utilities
│   └── requirements.txt  # Core python packages
├── frontend/             # React application
│   ├── src/              # App source code
│   │   ├── components/   # Shared UI components
│   │   ├── pages/        # Scan analysis, chat, symptom checker pages
│   │   └── utils/        # API and parsing hooks
│   └── tailwind.config.js
├── start-dev.ps1         # Local dev environment helper script
└── vercel.json           # Vercel deployment routing configuration
```

---

## ⚡ Vercel Deployment

This repository is optimized for deployment as a single Vercel project:
- **Frontend** compiles statically from `frontend/dist`.
- **Backend API** runs as a serverless Python function via the `api/` directory (matching `/api/*` requests).

### Environment Variables
For production deployment, add the following variables in your Vercel Project Dashboard:

| Variable Name | Type | Description |
| :--- | :--- | :--- |
| `GROQ_API_KEY` | Required | Your Groq API Credentials |
| `GROQ_ANALYZE_MODEL` | Optional | Vision-capable model (Default: `meta-llama/llama-4-scout-17b-16e-instruct`) |
| `GROQ_TEXT_MODEL` | Optional | General text-capable model (Default: `meta-llama/llama-4-scout-17b-16e-instruct`) |

### Deployment Steps
1. Connect your repository to **Vercel**.
2. Vercel automatically detects Vite configurations from `vercel.json`.
3. Add the `GROQ_API_KEY` under **Project Settings -> Environment Variables**.
4. Deploy!

---

## ⚠️ Medical Disclaimer

This application is an AI-assisted analysis tool meant for educational and informational purposes. It does not provide medical diagnoses, treatment recommendations, or professional clinical opinions. Always consult a licensed clinician for medical decisions.
