import cv2
import time
import os
import streamlit as st
import numpy as np
import base64
import speech_recognition as sr
from gtts import gTTS
import tempfile
from posturedetection import PostureDetector
from llama_cloud_services import LlamaParse
from groq import Groq
from dotenv import load_dotenv
from streamlit import secrets as st_secrets
from streamlit.errors import StreamlitSecretNotFoundError
import speechRecognition as speech_module

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(page_title="AI Mock Interview", layout="wide")

# Fetch API keys
try:
    LLAMA_API_KEY = st_secrets["LLAMA_API_KEY"]
except (KeyError, StreamlitSecretNotFoundError):
    LLAMA_API_KEY = os.getenv("LLAMA_API_KEY", "")
try:
    GROQ_API_KEY = st_secrets["GROQ_API_KEY"]
except (KeyError, StreamlitSecretNotFoundError):
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

if not LLAMA_API_KEY:
    st.sidebar.error("Missing LlamaParse API key.")
if not GROQ_API_KEY:
    st.sidebar.error("Missing Groq API key.")

# Initialize once
if 'detector' not in st.session_state:
    st.session_state.detector = PostureDetector()
if 'parser' not in st.session_state and LLAMA_API_KEY:
    st.session_state.parser = LlamaParse(api_key=LLAMA_API_KEY)
if 'groq' not in st.session_state and GROQ_API_KEY:
    st.session_state.groq = Groq(api_key=GROQ_API_KEY)

# Initialize session state variables
if 'questions' not in st.session_state:
    st.session_state.questions = []
if 'answers' not in st.session_state:
    st.session_state.answers = []
if 'emotions' not in st.session_state:
    st.session_state.emotions = []
if 'current_question_idx' not in st.session_state:
    st.session_state.current_question_idx = 0
if 'interview_active' not in st.session_state:
    st.session_state.interview_active = False
if 'interview_completed' not in st.session_state:
    st.session_state.interview_completed = False
if 'posture_samples' not in st.session_state:
    st.session_state.posture_samples = []
if 'last_posture_check' not in st.session_state:
    st.session_state.last_posture_check = 0
if 'feedback' not in st.session_state:
    st.session_state.feedback = []
if 'start_time' not in st.session_state:
    st.session_state.start_time = time.time()
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'evaluations' not in st.session_state:
    st.session_state.evaluations = []
if 'question_spoken' not in st.session_state:
    st.session_state.question_spoken = False
if 'answer_recorded' not in st.session_state:
    st.session_state.answer_recorded = False
if 'wait_period' not in st.session_state:
    st.session_state.wait_period = None

# Function to convert text to speech
def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
        temp_filename = fp.name
        tts.save(temp_filename)
    
    with open(temp_filename, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        audio_html = f"""
            <audio autoplay="true">
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)
    
    os.unlink(temp_filename)
    return True

# Enhanced speech recognition function with emotion analysis
def recognize_speech_with_emotion(timeout=30):
    st.info("Listening for your answer...")
    try:
        # Use the imported speech module to capture and analyze audio
        text, emotion = speech_module.capture_audio_and_analyze()
        if text is None or emotion is None:
            return "Could not understand the audio.", "unknown"
        return text, emotion
    except Exception as e:
        st.error(f"Error during speech recognition: {str(e)}")
        return f"Error during speech recognition: {str(e)}", "unknown"

# Function to evaluate an answer with emotion context
def evaluate_answer(question, answer, emotion):
    if not question or not answer or answer in ["No answer provided within the time limit.", 
                                              "Could not understand the audio.", 
                                              "Could not request results from speech recognition service."]:
        return "No proper answer was provided to evaluate."
    
    prompt = f"""
    Question: {question}
    Answer: {answer}
    Detected Emotion: {emotion}
    
    Evaluate the above interview answer based on:
    1. Relevance to the question
    2. Clarity and coherence
    3. Content quality and depth
    4. Emotional congruence (was the emotional tone appropriate for the answer?)
    5. Overall impression
    
    Provide a score between 1-10 and detailed feedback that acknowledges their emotional tone.
    """
    
    try:
        completion = st.session_state.groq.chat.completions.create(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            messages=[{"role":"user","content":prompt}],
            temperature=0.7,
            max_completion_tokens=512,
            top_p=1,
            stream=False
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Evaluation error: {str(e)}"

# Function to create comprehensive final evaluation
def generate_final_evaluation(questions, answers, emotions, posture_samples, evaluations):
    # Calculate posture metrics
    if posture_samples:
        good_posture_count = sum(1 for sample in posture_samples if sample['is_good'])
        total_samples = len(posture_samples)
        posture_percentage = (good_posture_count / total_samples * 100) if total_samples > 0 else 0
        common_issues = {}
        
        for sample in posture_samples:
            if not sample['is_good']:
                for issue in sample['issues']:
                    if issue in common_issues:
                        common_issues[issue] += 1
                    else:
                        common_issues[issue] = 1
        
        top_issues = sorted(common_issues.items(), key=lambda x: x[1], reverse=True)[:3]
    else:
        posture_percentage = 0
        top_issues = []
    
    # Analyze emotions
    emotion_counts = {}
    for emotion in emotions:
        if emotion != "unknown":
            if emotion in emotion_counts:
                emotion_counts[emotion] += 1
            else:
                emotion_counts[emotion] = 1
    
    top_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)

    # Build comprehensive evaluation prompt
    prompt = f"""
    You are an expert interview coach reviewing a mock interview session. Please provide a comprehensive evaluation of the candidate's performance.
    
    POSTURE ASSESSMENT:
    - Overall posture quality: {posture_percentage:.1f}%
    - Top posture issues: {', '.join([f"{issue} ({count} times)" for issue, count in top_issues]) if top_issues else "None detected"}
    
    EMOTIONAL ANALYSIS:
    - Emotions detected: {', '.join([f"{emotion} ({count} times)" for emotion, count in top_emotions]) if top_emotions else "No clear emotions detected"}
    
    QUESTION & ANSWER ASSESSMENT:
    {chr(10).join([f"Q{i+1}: {q}\nA: {a}\nEmotion: {e}\nEvaluation: {eval}" for i, (q, a, e, eval) in enumerate(zip(questions, answers, emotions, evaluations))])}
    
    Based on all this information, please provide:
    1. Overall interview performance score (1-10)
    2. Key strengths demonstrated
    3. Areas for improvement
    4. Specific advice on posture and non-verbal communication
    5. Specific advice on emotional expression and tone
    6. Recommendations for next practice steps
    
    Format your response as a professional coaching feedback session.
    """
    
    try:
        completion = st.session_state.groq.chat.completions.create(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            messages=[{"role":"user","content":prompt}],
            temperature=0.7,
            max_completion_tokens=1024,
            top_p=1,
            stream=False
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Final evaluation error: {str(e)}"

# Sidebar
with st.sidebar:
    st.header("Camera & Controls")
    
    # Live camera feed placeholder
    video_widget = st.empty()
    
    st.markdown("---")
    st.subheader("Interview Status")
    status_area = st.empty()
    
    # Add countdown timer display for automated process
    timer_display = st.empty()
    
    st.markdown("---")
    st.subheader("Recent Posture Feedback")
    posture_status = st.empty()
    
    st.markdown("---")
    st.subheader("Interview Setup")
    resume_file = st.file_uploader("Upload Resume", type=["pdf", "docx"])
    
    col1, col2 = st.columns(2)
    with col1:
        start_interview = st.button("Start Interview", disabled=st.session_state.interview_active)
    with col2:
        end_interview = st.button("End Interview", disabled=not st.session_state.interview_active)
    
    st.markdown("---")
    st.subheader("Settings")
    show_landmarks = st.checkbox("Show Pose Landmarks", value=True)
    posture_check_interval = st.slider("Posture Check Interval (seconds)", 5, 60, 15)
    recording_duration = st.slider("Answer Recording Duration (seconds)", 10, 120, 45)
    
    st.markdown("---")
    st.markdown("**Instructions:**\n- Upper body visible\n- Upright posture\n- Head aligned\n- Look at camera")

# Main content area
main_container = st.container()

with main_container:
    st.title("AI Mock Interview System")
    
    # Display current interview state
    if not st.session_state.interview_active and not st.session_state.interview_completed:
        st.info("Upload your resume and click 'Start Interview' to begin.")
    
    # Active interview section
    if st.session_state.interview_active:
        st.subheader("Current Question")
        q_container = st.container()
        a_container = st.container()
        
        # Initialize the current question state
        if 'question_spoken' not in st.session_state:
            st.session_state.question_spoken = False
        if 'answer_recorded' not in st.session_state:
            st.session_state.answer_recorded = False
        if 'wait_period' not in st.session_state:
            st.session_state.wait_period = None
        
        with q_container:
            current_q = st.session_state.questions[st.session_state.current_question_idx]
            st.write(f"**Q{st.session_state.current_question_idx + 1}:** {current_q}")
            status_msg = st.empty()
            
            # Automatically speak the question if not already done
            if not st.session_state.question_spoken:
                status_msg.info("Speaking question...")
                text_to_speech(current_q)
                st.session_state.question_spoken = True
                st.session_state.wait_period = time.time()
                st.rerun()
        
        with a_container:
            answer_status = st.empty()
            emotion_indicator = st.empty()
            
            # Give a short pause after speaking the question
            if st.session_state.question_spoken and not st.session_state.answer_recorded:
                current_time = time.time()
                
                # Wait 2 seconds after speaking before recording
                if st.session_state.wait_period and (current_time - st.session_state.wait_period) >= 2:
                    # Start recording the answer
                    if not st.session_state.recording:
                        st.session_state.recording = True
                        answer_status.warning("Recording your answer... (speak now)")
                        answer, emotion = recognize_speech_with_emotion(timeout=recording_duration)
                        st.session_state.answers[st.session_state.current_question_idx] = answer
                        st.session_state.emotions[st.session_state.current_question_idx] = emotion
                        answer_status.write(f"**Your Answer:** {answer}")
                        emotion_indicator.write(f"**Detected Emotion:** {emotion}")
                        st.session_state.recording = False
                        st.session_state.answer_recorded = True
                        
                        # Set a delay before moving to next question
                        st.session_state.wait_period = time.time()
                        st.rerun()
                else:
                    answer_status.info("Preparing to record your answer...")
            
            # After answer is recorded, prepare for next question
            if st.session_state.answer_recorded:
                current_time = time.time()
                
                # Wait 5 seconds after recording before next question
                if (current_time - st.session_state.wait_period) >= 5:
                    # Move to next question if not at the end
                    if st.session_state.current_question_idx < len(st.session_state.questions) - 1:
                        st.session_state.current_question_idx += 1
                        st.session_state.question_spoken = False
                        st.session_state.answer_recorded = False
                        st.rerun()
                    else:
                        st.session_state.interview_active = False
                        st.session_state.interview_completed = True
                        st.rerun()
                else:
                    remaining = 5 - (current_time - st.session_state.wait_period)
                    answer_status.info(f"Moving to next question in {int(remaining)} seconds...")
    
    # Interview completed section
    if st.session_state.interview_completed:
        st.success("Interview completed! Here's your evaluation.")
        
        # Generate evaluations if not already done
        if not st.session_state.evaluations:
            with st.spinner("Evaluating your answers..."):
                for q_idx, (question, answer, emotion) in enumerate(zip(
                        st.session_state.questions, st.session_state.answers, st.session_state.emotions)):
                    evaluation = evaluate_answer(question, answer, emotion)
                    st.session_state.evaluations.append(evaluation)
        
        # Generate comprehensive final evaluation
        final_eval_container = st.container()
        with final_eval_container:
            st.subheader("Comprehensive Interview Evaluation")
            
            with st.spinner("Generating comprehensive evaluation..."):
                final_evaluation = generate_final_evaluation(
                    st.session_state.questions,
                    st.session_state.answers,
                    st.session_state.emotions,
                    st.session_state.posture_samples,
                    st.session_state.evaluations
                )
                st.markdown(final_evaluation)
        
        # Display Q&A with evaluations
        st.subheader("Question-by-Question Analysis")
        for q_idx, (question, answer, emotion, evaluation) in enumerate(zip(
                st.session_state.questions, st.session_state.answers, 
                st.session_state.emotions, st.session_state.evaluations)):
            with st.expander(f"Question {q_idx + 1}", expanded=False):
                st.write(f"**Question:** {question}")
                st.write(f"**Your Answer:** {answer}")
                st.write(f"**Detected Emotion:** {emotion}")
                st.markdown("### Evaluation")
                st.write(evaluation)
        
        # Display posture analysis
        st.subheader("Posture Analysis Details")
        
        # Calculate posture metrics
        if st.session_state.posture_samples:
            good_posture_count = sum(1 for sample in st.session_state.posture_samples if sample['is_good'])
            total_samples = len(st.session_state.posture_samples)
            posture_percentage = (good_posture_count / total_samples * 100) if total_samples > 0 else 0
            
            st.metric("Overall Posture Quality", f"{posture_percentage:.1f}%")
            
            # Display posture timeline
            st.subheader("Posture Timeline")
            timeline_data = []
            for idx, sample in enumerate(st.session_state.posture_samples):
                timestamp = sample['timestamp']
                status = "✅ Good" if sample['is_good'] else "⚠️ Needs Improvement"
                issues = ", ".join(sample['issues']) if not sample['is_good'] else "None"
                timeline_data.append({"Time": f"{timestamp:.1f}s", "Status": status, "Issues": issues})
            
            for entry in timeline_data:
                st.text(f"[{entry['Time']}] {entry['Status']} - Issues: {entry['Issues']}")
        else:
            st.warning("No posture data was collected during the interview.")
        
        # Display emotion analysis
        st.subheader("Emotion Analysis")
        emotion_counts = {}
        for emotion in st.session_state.emotions:
            if emotion != "unknown":
                if emotion in emotion_counts:
                    emotion_counts[emotion] += 1
                else:
                    emotion_counts[emotion] = 1
                    
        if emotion_counts:
            # Create a bar chart of emotions
            emotion_data = [[emotion, count] for emotion, count in emotion_counts.items()]
            emotion_data.sort(key=lambda x: x[1], reverse=True)
            
            st.bar_chart(dict(emotion_counts))
            
            # Display emotion by question
            st.subheader("Emotions by Question")
            for i, (q, e) in enumerate(zip(st.session_state.questions, st.session_state.emotions)):
                st.text(f"Q{i+1}: {e}")
        else:
            st.warning("No emotions were detected during the interview.")
        
        # Reset button
        if st.button("Start New Interview"):
            # Reset session state
            st.session_state.questions = []
            st.session_state.answers = []
            st.session_state.emotions = []
            st.session_state.current_question_idx = 0
            st.session_state.interview_active = False
            st.session_state.interview_completed = False
            st.session_state.posture_samples = []
            st.session_state.last_posture_check = 0
            st.session_state.feedback = []
            st.session_state.start_time = time.time()
            st.session_state.evaluations = []
            st.session_state.question_spoken = False
            st.session_state.answer_recorded = False
            st.session_state.wait_period = None
            st.rerun()

# When interview starts, parse resume & generate questions
if start_interview and resume_file and LLAMA_API_KEY and GROQ_API_KEY:
    with st.spinner("Analyzing resume and generating interview questions..."):
        info = {"file_name": resume_file.name}
        job_res = st.session_state.parser.parse(resume_file, extra_info=info)
        text = getattr(job_res, "content", str(job_res))
        prompt = ("Based on the following resume, generate 10 - 12 relevant mock interview questions (and only questions and nothing else) "
                 "that would help evaluate the candidate's skills and experience:" + text)
        
        completion = st.session_state.groq.chat.completions.create(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            messages=[{"role":"user","content":prompt}],
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            stream=False
        )
        raw_q = completion.choices[0].message.content or ""
        
        # Split and clean questions
        qs = [
            line.strip(" -\t0123456789.") 
            for line in raw_q.splitlines() if line.strip()
        ]
        # Ensure all questions end with a question mark
        qs = [q if q.endswith("?") else f"{q}?" for q in qs if q]
        
        # Filter out non-questions
        qs = [q for q in qs if "?" in q]
        
        st.session_state.questions = qs
        st.session_state.answers = [""] * len(qs)
        st.session_state.emotions = ["unknown"] * len(qs)
        st.session_state.interview_active = True
        st.session_state.start_time = time.time()
        st.session_state.question_spoken = False
        st.session_state.answer_recorded = False
        st.session_state.wait_period = None
        st.rerun()

# End interview when requested
if end_interview:
    st.session_state.interview_active = False
    st.session_state.interview_completed = True
    st.rerun()

# Always capture video feed
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    st.sidebar.error("⚠️ Cannot access webcam")
else:
    ret, frame = cap.read()
    
    if ret:
        frame = cv2.flip(frame, 1)
        current_time = time.time() - st.session_state.start_time
        
        # Check posture at intervals when interview is active
        if (st.session_state.interview_active and 
            current_time - st.session_state.last_posture_check >= posture_check_interval):
            
            # Run posture detection
            proc_frame, feedback = st.session_state.detector.process_frame(frame)
            
            # Save posture sample
            is_good_posture = any("good" in msg.lower() for msg in feedback)
            issues = [msg for msg in feedback if "good" not in msg.lower()]
            
            posture_sample = {
                'timestamp': current_time,
                'is_good': is_good_posture,
                'issues': issues if not is_good_posture else []
            }
            st.session_state.posture_samples.append(posture_sample)
            
            # Update last check time
            st.session_state.last_posture_check = current_time
            
            # Save feedback for display
            st.session_state.feedback = feedback
            
            # Update display frame with landmarks if requested
            display_frame = proc_frame if show_landmarks else frame
        else:
            # Just display the frame without analysis
            display_frame = frame
        
        # Show feed in sidebar
        video_widget.image(display_frame, channels="BGR", use_container_width=True)
        
        # Show interview status
        with status_area.container():
            if st.session_state.interview_active:
                st.success(f"Interview in progress - Question {st.session_state.current_question_idx + 1} of {len(st.session_state.questions)}")
                st.info(f"Time elapsed: {int(current_time)}s")
                if st.session_state.recording:
                    st.warning("Recording your answer...")
                elif st.session_state.question_spoken and not st.session_state.answer_recorded:
                    st.info("Getting ready to record your answer...")
                elif st.session_state.answer_recorded:
                    st.info("Preparing next question...")
            elif st.session_state.interview_completed:
                st.success("Interview completed!")
            else:
                st.info("Interview not started")
        
        # Display timer for automated process
        with timer_display.container():
            if st.session_state.interview_active:
                if st.session_state.wait_period and st.session_state.question_spoken and not st.session_state.answer_recorded:
                    wait_time = time.time() - st.session_state.wait_period
                    if wait_time < 2:
                        st.info(f"Starting to record in {2 - int(wait_time)} seconds...")
                elif st.session_state.wait_period and st.session_state.answer_recorded:
                    wait_time = time.time() - st.session_state.wait_period
                    if wait_time < 5:
                        st.info(f"Next question in {5 - int(wait_time)} seconds...")
        
        # Show recent posture feedback
        with posture_status.container():
            if st.session_state.feedback:
                for msg in st.session_state.feedback:
                    if "good" in msg.lower():
                        st.success(msg)
                    elif "warn" in msg.lower() or "detected" in msg.lower():
                        st.warning(msg)
                    else:
                        st.info(msg)
            else:
                st.info("Posture analysis will be shown here")
    
    cap.release()

# Footer
st.markdown("---")
st.caption("AI Mock Interview System — Posture Assessment, Emotion Analysis & Voice-based Q&A")