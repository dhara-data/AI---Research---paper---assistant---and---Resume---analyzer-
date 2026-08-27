import streamlit as st
import fitz
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq

# ==================================================
# GROQ CLIENT SETUP
# ==================================================
client = Groq(api_key=st.secrets["GROQ_API_KEY"])
GROQ_MODEL = "llama-3.1-8b-instant"  # fast + free tier. Use "llama-3.3-70b-versatile" for higher quality.


def ask_groq(system_prompt, user_prompt):
    """Helper to call Groq chat completion and return plain text."""
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content


def extract_pdf_text(uploaded_file):
    pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in pdf:
        text += page.get_text()
    pdf.close()
    return text


st.title("📚 AI Research Paper Assistant")

feature = st.sidebar.radio(
    "Choose a Feature",
    [
        "📚 Research Paper Assistant",
        "📋 Resume Analyzer"
    ]
)

# ==================================================
# RESEARCH PAPER ASSISTANT
# ==================================================
if feature == "📚 Research Paper Assistant":

    uploaded_file = st.file_uploader("Upload Research Paper (PDF)", type="pdf")

    if uploaded_file is not None:

        # Read PDF
        text = extract_pdf_text(uploaded_file)

        # Split text into chunks
        chunk_size = 500
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i + chunk_size])

        st.success("PDF uploaded successfully!")

        # Load Embedding Model
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Create Embeddings
        embeddings = embedding_model.encode(chunks)
        embedding_array = np.array(embeddings).astype("float32")

        # Create FAISS Index
        index = faiss.IndexFlatL2(embedding_array.shape[1])
        index.add(embedding_array)

        st.success("FAISS Index Created Successfully!")

        # ------------------------------------------
        # ASK A QUESTION
        # ------------------------------------------
        question = st.text_input("Ask a question about the research paper")

        if question:
            question_embedding = embedding_model.encode([question]).astype("float32")
            distances, indices = index.search(question_embedding, k=1)
            best_chunk = chunks[indices[0][0]]

            st.subheader("Most Relevant Chunk")
            st.write(best_chunk)

            with st.spinner("Thinking..."):
                answer = ask_groq(
                    system_prompt="You are a helpful AI Research Paper Assistant. Answer in simple English.",
                    user_prompt=f"""
Research Paper:

{best_chunk}

Question:

{question}

Answer in simple language so that even a beginner can understand.
"""
                )

            st.subheader("Answer")
            st.write(answer)

        st.divider()

        # ------------------------------------------
        # SUMMARY
        # ------------------------------------------
        st.subheader("📝 Research Paper Summary")

        if st.button("Generate Summary"):
            summary_text = text[:12000]

            with st.spinner("Generating summary..."):
                summary = ask_groq(
                    system_prompt="You summarize academic research papers in simple English.",
                    user_prompt=f"""
Summarize the following research paper.

Include:

1. Research objective
2. Problem being solved
3. Methodology
4. Dataset
5. Main findings
6. Results
7. Limitations
8. Conclusion

Use simple English.

Research Paper:

{summary_text}
"""
                )

            st.write(summary)

        # ------------------------------------------
        # KEY POINTS
        # ------------------------------------------
        st.divider()
        st.subheader("📌 Key Points")

        if st.button("Extract Key Points"):
            key_points_text = text[:12000]

            with st.spinner("Extracting key points..."):
                key_points = ask_groq(
                    system_prompt="""
You are an academic research assistant.

Extract the most important information
from research papers.
""",
                    user_prompt=f"""
Read the following research paper
and extract its key points.

Organize the answer into:

📌 Research Objective

📌 Problem Statement

📌 Methodology

📌 Dataset

📌 Important Findings

📌 Results

📌 Limitations

📌 Conclusion

Use bullet points and simple English.

Research Paper:

{key_points_text}
"""
                )

            st.write(key_points)


# ==================================================
# RESUME ANALYZER
# ==================================================
elif feature == "📋 Resume Analyzer":

    st.header("📋 AI Resume Analyzer")

    st.write(
        "Upload your resume and compare it with "
        "the requirements of your target job."
    )

    resume_file = st.file_uploader("Upload Resume (PDF)", type="pdf")

    job_role = st.text_input(
        "🎯 Enter Target Job Role",
        placeholder="Example: Data Scientist"
    )

    job_description = st.text_area(
        "📄 Paste Job Description",
        placeholder="""
Paste the job description here.

Example:

We are looking for a Data Scientist
with experience in Python, SQL,
Machine Learning, Pandas, NumPy,
Power BI and Statistics.
"""
    )

    if st.button("🔍 Analyze Resume"):

        if resume_file is None:
            st.warning("Please upload your resume.")
            st.stop()

        if not job_role:
            st.warning("Please enter the target job role.")
            st.stop()

        with st.spinner("Reading resume..."):
            resume_text = extract_pdf_text(resume_file)

        if not resume_text.strip():
            st.error("Could not extract text from resume.")
            st.stop()

        if job_description.strip():
            job_info = job_description
        else:
            job_info = f"""
Target Job Role:

{job_role}

Analyze the common requirements,
skills and qualifications normally
expected for this role.
"""

        with st.spinner("AI is analyzing your resume..."):
            analysis = ask_groq(
                system_prompt="""
You are an expert resume and career
analysis assistant.

Analyze the resume against the
target job role.

Be honest and practical.

Identify:

1. Skills already present
2. Missing skills
3. Missing projects
4. Missing achievements
5. Missing experience
6. Resume weaknesses
7. Recommended improvements

Use simple English.
""",
                user_prompt=f"""
TARGET JOB:

{job_role}


JOB DESCRIPTION:

{job_info}


RESUME:

{resume_text[:15000]}


Analyze the resume and provide:

## Skills Already Present

## Missing Skills

## Missing Projects

## Missing Achievements

## Missing Experience

## Resume Weaknesses

## Recommended Improvements

## Overall Recommendation

Give practical recommendations
that the student can actually follow.
"""
            )

        st.subheader("🤖 Resume Analysis")
        st.write(analysis)
