import os
import spacy
import re
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the English NLP model
nlp = spacy.load("en_core_web_sm")

# Function to load files
def load_files(file_paths):
    """Load multiple text files and return their contents."""
    contents = {}
    for file_path in file_paths:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read().strip()
                if not content:
                    print(f"Warning: {file_path} is empty!")
                contents[file_path] = content
        else:
            print(f"Error: {file_path} does not exist!")
    return contents

# Load multiple resumes and job descriptions
resume_files = ['resume_manish.txt','Raj_resume.txt']  # Your resume file
jd_files = ['jd_software_developer.txt']  # Your job description files





# Load content into dictionaries
resumes = load_files(resume_files)
jds = load_files(jd_files)

# List to hold results
results = []

# Process each resume and job description
for resume_name, resume_text in resumes.items():
    for jd_name, jd_text in jds.items():
        print(f"Processing {resume_name} against {jd_name}...")

        # Step 2: Preprocess Text using spaCy
        def preprocess(text):
            """Preprocess text by removing stop words and punctuation."""
            doc = nlp(text)
            tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
            return tokens

        # Preprocess resume and JD text
        resume_tokens = preprocess(resume_text)
        jd_tokens = preprocess(jd_text)

        # Step 3: Extract Features (Skills, Experience, Education)
        skills_list = ['Python', 'JavaScript', 'React', 'Machine Learning', 'Data Science']

        # Function to extract skills from text
        def extract_skills(text, skills):
            """Extract specified skills from the text."""
            extracted_skills = [skill for skill in skills if skill.lower() in text.lower()]
            return extracted_skills

        # Extract skills from resume and JD
        resume_skills = extract_skills(resume_text, skills_list)
        jd_skills = extract_skills(jd_text, skills_list)

        # Function to extract years of experience
        def extract_experience(text):
            """Extract years of experience from text."""
            experience = re.findall(r'(\d+) years', text)
            return max(map(int, experience), default=0)

        # Function to extract education level
        def extract_education(text):
            """Extract education level from text."""
            if 'B.Tech' in text or 'Bachelor' in text:
                return 'Bachelors'
            elif 'Masters' in text or 'M.Tech' in text:
                return 'Masters'
            else:
                return 'Unknown'

        # Extract experience and education from resume and JD
        resume_experience = extract_experience(resume_text)
        jd_experience = extract_experience(jd_text)

        resume_education = extract_education(resume_text)
        jd_education = extract_education(jd_text)

        # Step 4: Build Feature Vector for Machine Learning
        def build_feature_vector(resume_skills, jd_skills, resume_experience, jd_experience, resume_education, jd_education):
            """Build feature vector based on extracted features."""
            matching_skills = len(set(resume_skills).intersection(set(jd_skills)))  # Number of matching skills
            experience_match = abs(resume_experience - jd_experience)  # Difference in experience
            education_match = 1 if resume_education == jd_education else 0  # Check if education matches
            return np.array([matching_skills, experience_match, education_match])

        # Build feature vector for this resume
        features = build_feature_vector(resume_skills, jd_skills, resume_experience, jd_experience, resume_education, jd_education)

        # Adding a description of matching criteria
        description = f"Matching Skills: {features[0]}, Experience Difference: {features[1]}, Education Match: {features[2]}"

        # Step 5: Train the Machine Learning Model (Random Forest)
        # Sample dataset (features for multiple resumes)
        X = np.array([[3, 1, 1], [2, 0, 1], [1, 3, 0], [4, 0, 1]])  # Example feature vectors
        y = np.array([1, 1, 0, 1])  # 1 = relevant, 0 = not relevant

        # Check if we have enough samples to train
        if len(X) > 1:
            # Split into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train Random Forest model
            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            # Test the model
            accuracy = model.score(X_test, y_test)
            print(f"Model Accuracy: {accuracy * 100:.2f}%")

            # Predict the relevance of a new resume (use the feature vector we built)
            prediction = model.predict([features])

            # Append results
            results.append({
                "CV File Name": resume_name,
                "JD File Name": jd_name,
                "Score": prediction[0],  # Display the raw prediction (0 or 1)
                "Feature Vector": features.tolist(),  # Convert to list for better readability
                "Description": description  # Append the description
            })
        else:
            print("Not enough samples to perform train-test split.")

# Print results table
print("\nResults:")
print(f"{'CV File Name':<20} {'JD File Name':<25} {'Score':<10} {'Feature Vector':<15} {'Description'}")
for result in results:
    print(f"{result['CV File Name']:<20} {result['JD File Name']:<25} {result['Score']:<10} {str(result['Feature Vector']):<15} {result['Description']}")

print("Legend")
print("0 Means Non Relevant")
print("1 Means Relevant")