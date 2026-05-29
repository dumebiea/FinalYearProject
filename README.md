# Food Crisis Detection System

## Project Overview
The Food Crisis Detection System is an AI-powered platform designed to monitor food security in Lagos, Nigeria. By analyzing social media discussions (Twitter/X) using advanced Natural Language Processing (NLP), the system detects early signs of food scarcity, price hikes, and hunger crises. It provides real-time sentiment analysis, spatial visualizations (heatmaps), and automated alerts to help stakeholders make informed decisions.

## Prerequisites
- Python 3.10+
- Node.js 18+
- npm

## Project Structure
```text
food-crisis-sentiment-system/
├── backend/                # Flask API & AI Services
│   ├── models/             # Database Schemas
│   ├── routes/             # API Endpoints
│   ├── services/           # NLP, Summary & Scheduler Logic
│   └── instance/           # SQLite Database
├── frontend/               # React Dashboard
│   ├── src/components/     # UI Components & Charts
│   └── src/pages/          # Dashboard Views
├── data/                   # Dataset for collection simulation
├── model/                  # Pre-trained BERT & SVM models
└── requirements.txt        # Python dependencies
```

## Setup Instructions

### Step 1: Clone and navigate
`cd food-crisis-sentiment-system`

### Step 2: Activate virtual environment
**Windows:**
`venv\Scripts\activate`

**Mac/Linux:**
`source venv/bin/activate`

### Step 3: Install backend dependencies
`cd backend`
`pip install -r requirements.txt`

### Step 4: Configure email (optional)
Open `backend/config.py`
Replace `MAIL_USERNAME` and `MAIL_PASSWORD` with your Gmail credentials.
To get a Gmail app password: **Google Account → Security → App Passwords**.

### Step 5: Start the backend
`cd backend`
`python app.py`
*Server starts at http://localhost:5000*
*First startup takes 1-2 minutes (loading AI model)*

### Step 6: Start the frontend (new terminal)
`cd frontend`
`npm install`
`npm start`
*Opens at http://localhost:3000*

## Default Login Credentials
- **Admin**:  `admin@foodcrisis.ng`  / `Admin@123`
- **Viewer**: `viewer@foodcrisis.ng` / `Viewer@123`

## Features
- **Real-time Tweet Classification**: Instant sentiment analysis of individual food-related tweets.
- **CSV Data Processing**: Batch upload and analysis of historical or bulk tweet data.
- **Dynamic Dashboard**: Interactive charts showing sentiment trends and tweet volumes.
- **Crisis Alert System**: Automated email notifications when negative sentiment exceeds thresholds.
- **Lagos LGA Heatmap**: Visual representation of food security status across Lagos LGAs.
- **Role-Based Access**: Secure login for Admin (full access) and Viewer (read-only) roles.

## Model Information
The system uses **NaijaSenti**, a transformer-based BERT model specifically trained on Nigerian Twitter data for high accuracy in local contexts. A **Support Vector Machine (SVM)** model serves as a robust fallback for legacy environments.

## Alert Thresholds
- **Warning**: Negative sentiment > 35%
- **Crisis**: Negative sentiment > 50%

---
*Developed for Food Crisis Detection, Covenant University 2025*