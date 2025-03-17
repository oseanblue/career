AI Career Coach

Overview

AI Career Coach is a tool leveraging Meta-Llama-3-8B-Instruct to assist users in optimizing their resumes, exploring career paths, and preparing for job interviews.

Features

1.Resume Optimization: Enhances resume content based on the desired job role.

2.Career Guidance: Suggests career paths based on skills and experience.

3.Mock Interview: Generates interview questions and tips for specific roles.

4.Interview Feedback: Evaluates user responses and suggests improvements.

How It Leverages Llama 3

The AI agent uses the Meta-Llama-3-8B-Instruct model for text generation, optimized for low VRAM using 4-bit quantization. It generates refined resumes, career insights, interview questions, and feedback in an interactive format.

Setup & Execution

Prerequisites

Python 3.8+

PyTorch

Transformers library (Hugging Face)

BitsAndBytes for 4-bit quantization

Installation

Clone the repository or extract the provided ZIP file.

git clone https://github.com/your-repo/career.git
cd career

Install dependencies:

pip install torch transformers bitsandbytes

Run the script:

python main.py

Usage

Select an option:

1: Optimize Resume

2: Career Guidance

3: Mock Interview

4: Exit

Follow prompts to input resume details, skills, or job roles.

Receive AI-generated responses and guidance.

Dependencies

1.torch

2.transformers

3.bitsandbytes

Notes

Model loading may take a few minutes.

Ensure GPU support for faster execution.



