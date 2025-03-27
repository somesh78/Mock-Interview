import os
from dotenv import load_dotenv
from groq import Groq
from docx import Document
import spacy
from fpdf import FPDF
import re
from speechRecognition import *
from posturedetection import *
from datetime import datetime

load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
client = Groq(api_key=GROQ_API_KEY)

nlp = spacy.load("en_core_web_sm")

def extract_resume_info(resume_path):
    doc = Document(resume_path)
    resume_text = ""
    for para in doc.paragraphs:
        resume_text += para.text + "\n"
    
    return extract_projects_and_skills(resume_text)

def extract_projects_and_skills(text, personal_info=None, skill_keywords=None, project_keywords=None):
    if skill_keywords is None:
        skill_keywords = ["machine learning", "deep learning", "python", "data analysis", "AI", "neural networks"]
    
    if project_keywords is None:
        project_keywords = ["project", "developed", "implemented", "designed", "built"]

    if personal_info is None:
        personal_info = []

    for info in personal_info:
        text = re.sub(re.escape(info), "", text)

    skills = []
    projects = []
    
    for line in text.split("\n"):
        line = line.strip()
        
        if any(skill.lower() in line.lower() for skill in skill_keywords):
            skills.append(line)

        if any(project.lower() in line.lower() for project in project_keywords):
            projects.append(line)
    
    return skills, projects

def generate_custom_questions(projects, skills):
    questions = []
    
    for project in projects:
        questions.append(f"Can you tell me more about your experience working on {project}? What was your role?")
    
    for skill in skills:
        questions.append(f"How have you applied {skill} in your previous projects?")
    
    return questions

def get_feedback(user_answer, correct_answer):
    prompt = f"The user provided the following answer: {user_answer}. Compare it with this correct answer: {correct_answer}. Provide feedback."
    
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama3-8b-8192",
    )

    return chat_completion.choices[0].message.content

def get_resume_feedback(resume_text):
    prompt = f"Based on this resume content: '{resume_text}', provide feedback on the structure, clarity, skills, and any potential areas of improvement."
    
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama3-8b-8192",
    )

    return chat_completion.choices[0].message.content

def generate_pdf(questions, pdf_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Interview Questions", ln=True, align="C")

    for i, question in enumerate(questions, start=1):
        pdf.multi_cell(0, 10, txt=f"Question {i}: {question}")

    pdf.output(pdf_path)

posture_data = {}

def record_posture_data(posture, eye_movement):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    posture_data[timestamp] = {
        "posture": posture,
        "eye_movement": eye_movement
    }

def conduct_interview(resume_path, pdf_path):
    print("Welcome to the Personalized Interview Bot!\n")

    personal_info = ["Your Name", "your.email@example.com", "Your University"]
    skills, projects = extract_resume_info(resume_path)

    if not skills and not projects:
        print("No skills or projects detected in the resume. The interview will proceed with general questions based on your resume.")
        questions = []
    else:
        questions = generate_custom_questions(projects, skills)

    all_questions = []
    user_answers = []
    correct_answers = []

    for index, question in enumerate(questions, start=1):
        print(f"Question {index}: {question}")
        all_questions.append(question)

        user_answer = input("Your answer: ")
        user_answers.append(user_answer)

        posture_feedback, eye_movement = assess_posture(results.pose_landmarks.landmark)
        record_posture_data(posture_feedback, eye_movement)
        print("Posture Analysis Result:", posture_feedback)

        _, tone_result = capture_audio_and_analyze()
        if tone_result:
            print(f"Tone Analysis Result: You seem to be expressing a {tone_result} tone.")

        correct_answer_prompt = f"Provide a detailed explanation for: {question}"
        correct_answer = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": correct_answer_prompt,
                }
            ],
            model="llama3-8b-8192",
        ).choices[0].message.content
        correct_answers.append(correct_answer)

    print("Interview session ended. Thank you!\n")

    overall_feedback = []
    for user_answer, correct_answer in zip(user_answers, correct_answers):
        feedback = get_feedback(user_answer, correct_answer)
        overall_feedback.append(feedback)

    print("Overall Feedback for Interview:\n")
    for idx, feedback in enumerate(overall_feedback, start=1):
        print(f"Feedback for Question {idx}: {feedback}\n")

    print("Overall Posture and Eye Movement Data:\n")
    for timestamp, data in posture_data.items():
        print(f"At {timestamp}: Posture - {data['posture']}, Eye Movement - {data['eye_movement']}")

    try:
        with open(resume_path, "r", encoding="utf-8") as file:
            resume_text = file.read()
    except UnicodeDecodeError as e:
        print(f"Error reading file: {e}")
        with open(resume_path, "r", encoding="latin-1") as file:
            resume_text = file.read()

    resume_feedback = get_resume_feedback(resume_text)
    print("Resume Feedback:\n")
    print(resume_feedback)

    generate_pdf(all_questions, pdf_path)
    print(f"The interview questions have been saved in {pdf_path}.")

if __name__ == "__main__":
    resume_path = input("Please enter the path to your resume: ")
    pdf_path = "interview_questions.pdf"
    conduct_interview(resume_path, pdf_path)
