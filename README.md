# Vignan AI Counselor 

A professional multi-agent AI system built to provide academic guidance, emotional support, and career counseling for Vignan University students.

 Features
Agentic Framework: Uses LangGraph for intelligent query routing.
Academic Support:Answers queries regarding Attendance, Exam Policies, and Fees.
Wellness Hub: Provides empathetic support for stress and emotional well-being.
Career Pathing:Analyzes student resumes to provide personalized career roadmaps.
Voice-Enabled:Supports voice consultation using Whisper and gTTS.

Tech Stack
Frontend:Streamlit
Brain:Llama 3.1 (via Groq)
Orchestration:LangChain & LangGraph
Vector Store:FAISS
Embeddings:HuggingFace (all-MiniLM-L6-v2)

Local Setup
1. Clone the repository.
2. Create a `.streamlit/secrets.toml` file and add your `GROQ_API_KEY`.
3. Install dependencies:
   --bash
   pip install -r requirements.txt