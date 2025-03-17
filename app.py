
import os
import torch
import fitz  # PyMuPDF for PDF reading
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

# Set environment variables to optimize CUDA memory
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Model name
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)

# Configure quantization to optimize for Kaggle GPUs
quant_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# Load model with memory limits
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,  
    device_map="sequential",
    torch_dtype=torch.float16,
    max_memory={0: "12GiB"}
)

# Text generation pipeline (Optimized)
text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=300,
    temperature=0.7,
    top_k=50,
    repetition_penalty=1.2
)

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text.strip()

# Function to refine resume
def refine_resume(resume_text, job_role):
    prompt = f"""
    You are a professional resume writer. Improve and refine the following resume for the role of {job_role}.
    - Ensure ATS (Applicant Tracking System) optimization.
    - Highlight relevant skills and achievements.
    - Remove unnecessary details.

    Resume Text:
    {resume_text}
    """
    response = text_generator(prompt, max_new_tokens=300)[0]['generated_text']
    return response

# Function to suggest best career fields based on skills & experience
def suggest_career(skills, experience):
    prompt = f"""
    You are a career advisor. Based on the following skills and years of experience, suggest the best career fields:
    - Skills: {skills}
    - Years of Experience: {experience}
    - Consider growing industries and high-paying opportunities.
    """
    response = text_generator(prompt, max_new_tokens=300)[0]['generated_text']
    return response

# Function for mock interview questions
def generate_mock_interview_questions(job_role):
    prompt = f"""
    You are an expert interviewer. Generate 5 important interview questions for the role of {job_role}.
    """
    response = text_generator(prompt, max_new_tokens=300)[0]['generated_text']
    return response

# Function to provide feedback on interview answers
def provide_interview_feedback(answer):
    prompt = f"""
    You are an HR professional. Evaluate the following interview answer and provide feedback for improvement:
    Answer: "{answer}"
    """
    response = text_generator(prompt, max_new_tokens=300)[0]['generated_text']
    return response

# Streamlit UI
st.title("📄 AI-Powered Career Assistant")
st.write("Upload your resume, get career suggestions, and practice for your next interview!")

# Section 1: Resume Upload & Refinement
st.subheader("📂 Upload Your Resume (PDF)")
uploaded_file = st.file_uploader("Choose a file", type="pdf")

if uploaded_file:
    job_role = st.text_input("Enter the job role you're applying for:")

    if job_role:
        with st.spinner("Extracting and refining resume..."):
            resume_text = extract_text_from_pdf(uploaded_file)
            refined_resume = refine_resume(resume_text, job_role)
        
        st.subheader("📄 Optimized Resume:")
        st.text_area("Your refined resume:", refined_resume, height=300)

# Section 2: Career Advice Based on Skills & Experience
st.subheader("💡 Career Path Suggestions")
skills = st.text_input("Enter your top skills (comma-separated):")
experience = st.number_input("Enter years of experience:", min_value=0, max_value=50, step=1)

if st.button("🔍 Get Career Suggestions"):
    if skills and experience >= 0:
        with st.spinner("Analyzing best career fields..."):
            career_suggestions = suggest_career(skills, experience)
        
        st.subheader("📌 Suggested Career Paths:")
        st.text_area("Best fields for you:", career_suggestions, height=200)

# Section 3: Mock Interview Practice
st.subheader("🎤 Mock Interview Practice")
job_role_interview = st.text_input("Enter job role for interview questions:")

if st.button("🎙 Generate Interview Questions"):
    if job_role_interview:
        with st.spinner("Generating interview questions..."):
            interview_questions = generate_mock_interview_questions(job_role_interview)
        
        st.subheader("📝 Interview Questions:")
        st.text_area("Prepare for these questions:", interview_questions, height=200)

# Section 4: AI Feedback on Interview Answers
st.subheader("🔍 AI Feedback on Your Answers")
user_answer = st.text_area("Enter your interview answer for evaluation:")

if st.button("📝 Get Feedback"):
    if user_answer:
        with st.spinner("Analyzing answer..."):
            feedback = provide_interview_feedback(user_answer)
        
        st.subheader("📢 AI Feedback:")
        st.text_area("Your answer feedback:", feedback, height=200)

