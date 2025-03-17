import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

# Enable efficient CUDA memory allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Model name
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

# Load tokenizer
print("[INFO] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)

# Configure 4-bit quantization (Optimized for Low VRAM)
quant_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# Load model with memory limits
print("[INFO] Loading model (this may take a few minutes)...")
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
    max_new_tokens=200,  # Allow longer responses
    temperature=0.7,
    top_k=50,
    repetition_penalty=1.2
)

# Function to refine and optimize resume
def refine_resume(resume_text, job_role):
    prompt = f"""
    Improve and optimize the following resume for a {job_role} position:
    {resume_text}
    - Highlight key achievements
    - Make it ATS (Applicant Tracking System) friendly
    - Use action-oriented language
    - Ensure it aligns with {job_role} job requirements
    - Provide missing skills or certifications to add
    """
    optimized_resume = text_generator(prompt)[0]['generated_text']
    return optimized_resume

# Function to suggest best career path based on skills
def career_guidance(skills, experience):
    prompt = f"""
    Based on these skills and experience, suggest the best career paths:
    Skills: {skills}
    Experience: {experience} years
    - Identify top 3 career paths
    - Mention industries that are growing
    - Suggest additional skills or certifications to increase opportunities
    """
    career_suggestions = text_generator(prompt)[0]['generated_text']
    return career_suggestions

# Function for mock interview
def mock_interview(job_role):
    prompt = f"""
    Generate 5 common interview questions for a {job_role} position.
    Also, provide tips on how to answer each question effectively.
    """
    interview_questions = text_generator(prompt)[0]['generated_text']
    return interview_questions

# Function to provide feedback on user answers
def interview_feedback(question, user_answer):
    prompt = f"""
    Evaluate the following interview answer and provide feedback:
    Question: {question}
    Answer: {user_answer}
    - Highlight strengths
    - Identify areas for improvement
    - Suggest how to make the answer more structured and compelling
    """
    feedback = text_generator(prompt)[0]['generated_text']
    return feedback

# Run in Terminal
if __name__ == "__main__":
    while True:
        print("\n💼 Career & Resume Optimizer")
        print("1️⃣ Optimize Resume")
        print("2️⃣ Career Guidance")
        print("3️⃣ Mock Interview")
        print("4️⃣ Exit")
        
        choice = input("Choose an option (1-4): ").strip()

        if choice == "1":
            print("\nPaste your resume text (large field allowed). Type END when done:")
            resume_lines = []
            while True:
                line = input()
                if line.strip().upper() == "END":
                    break
                resume_lines.append(line)
            resume_text = "\n".join(resume_lines)
            
            job_role = input("Enter the job role you are applying for: ").strip()
            print("\n[INFO] Optimizing your resume...\n")
            optimized_resume = refine_resume(resume_text, job_role)
            print("\n🔹 Optimized Resume:\n", optimized_resume)
        
        elif choice == "2":
            skills = input("\nEnter your top skills (comma separated): ").strip()
            experience = input("Enter years of experience: ").strip()
            print("\n[INFO] Analyzing best career paths...\n")
            career_suggestions = career_guidance(skills, experience)
            print("\n📈 Career Guidance:\n", career_suggestions)
        
        elif choice == "3":
            job_role = input("\nEnter the job role for the interview: ").strip()
            print("\n[INFO] Generating interview questions...\n")
            interview_questions = mock_interview(job_role)
            print("\n🎤 Mock Interview Questions:\n", interview_questions)

            for _ in range(3):
                print("\nPress ENTER to continue or type 'MENU' to go back.")
                back = input().strip().upper()
                if back == "MENU":
                    break
                
                question = input("\nEnter a question from the list: ").strip()
                user_answer = input("Enter your answer: ").strip()
                print("\n[INFO] Evaluating your response...\n")
                feedback = interview_feedback(question, user_answer)
                print("\n📝 Feedback on your answer:\n", feedback)
        
        elif choice == "4":
            print("\n👋 Exiting... Good luck with your career!")
            break
        
        else:
            print("❌ Invalid choice. Please try again.")
