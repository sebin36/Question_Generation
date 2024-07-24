import streamlit as st
import pickle
import numpy as np
import random

# Load the model and vectorizer
with open('question_generation_model.pkl', 'rb') as f:
    model, vectorizer, label_to_question = pickle.load(f)

# Streamlit app layout
st.title("Question Generator")
st.write("Enter some context text below, and the model will generate questions based on it.")

# Text input
context = st.text_area("Context:")

# Options for question type and number of questions
question_type = st.selectbox("Select Question Type:", ["Descriptive", "Multiple-Choice"])
num_questions = st.slider("Number of Questions:", 1, 10, 1)

def generate_descriptive_questions(context, num_questions):
    # Tokenize and extract key phrases or sentences
    # For simplicity, we'll split the context into sentences and use them directly.
    sentences = context.split('. ')
    if len(sentences) < num_questions:
        sentences.extend([''] * (num_questions - len(sentences)))  # Pad if not enough sentences
    random.shuffle(sentences)  # Shuffle to get diverse sentences
    
    questions = [f"What can you tell me about: '{sentences[i]}'?" for i in range(num_questions)]
    return questions

# Generate questions button
if st.button("Generate Questions"):
    if context:
        with st.spinner("Generating questions..."):
            # Transform the context using the vectorizer
            context_tfidf = vectorizer.transform([context])
            
            # Initialize list to store generated questions
            generated_questions = []
            
            # Set to track seen questions to avoid duplicates
            seen_questions = set()
            
            while len(generated_questions) < num_questions:
                # Predict the label
                predicted_label = model.predict(context_tfidf)[0]
                
                # Get the corresponding question
                question = label_to_question[predicted_label]
                
                if question not in seen_questions:
                    # Add the question to the list and set
                    generated_questions.append(question)
                    seen_questions.add(question)
                
                # Add some variation in question generation
                if question_type == "Multiple-Choice":
                    # Generate dummy options for MCQ (placeholder logic)
                    options = [question] + [f"{question} option {i+1}" for i in range(3)]
                    generated_questions.append({
                        'question': question,
                        'options': options
                    })
                elif question_type == "Descriptive":
                    generated_questions = generate_descriptive_questions(context, num_questions)


        st.write("Generated Questions:")
        for i, q in enumerate(generated_questions):
            if isinstance(q, dict):  # For MCQ
                st.write(f"{i + 1}: {q['question']}")
                for option in q['options']:
                    st.write(f"   - {option}")
            else:  # For descriptive
                st.write(f"{i + 1}: {q}")
    else:
        st.write("Please enter some context text.")
