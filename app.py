import streamlit as st
import fitz
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import groq import Groq

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


# -----------------------------
# Function to Extract PDF Text
# -----------------------------
def extract_pdf_text(uploaded_file):
    pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")

    text = ""

    for page in pdf:
        text += page.get_text()

    pdf.close()

    return text


# -----------------------------
# Title
# -----------------------------
st.title("📚 AI Research Paper Assistant")

# -----------------------------
# Sidebar
# -----------------------------
feature = st.sidebar.radio(
    "Choose a Feature",
    [
        "📚 Research Paper Assistant",
        "📋 Resume Analyzer"
    ]
)

if feature == "📚 Research Paper Assistant":

    uploaded_file = st.file_uploader(
        "Upload Research Paper (PDF)",
        type="pdf"
    )

    if uploaded_file is not None:

        # Extract text
        text = extract_pdf_text(uploaded_file)

        if not text.strip():
            st.error("Could not extract text from PDF.")
            st.stop()

        # Split into chunks
        chunk_size = 500
        chunks = []

        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i + chunk_size])

        st.success("PDF uploaded successfully!")

        # Load embedding model
        embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        # Create embeddings
        embeddings = embedding_model.encode(chunks)

        embedding_array = np.array(
            embeddings
        ).astype("float32")

        # Create FAISS index
        index = faiss.IndexFlatL2(
            embedding_array.shape[1]
        )

        index.add(embedding_array)

        st.success("FAISS Index Created Successfully!")

        # Ask Question
        question = st.text_input(
            "Ask a question about the research paper"
        )

        if question:

            question_embedding = embedding_model.encode(
                [question]
            ).astype("float32")

            distances, indices = index.search(
                question_embedding,
                k=1
            )

            best_chunk = chunks[
                indices[0][0]
            ]

            response = ollama.chat(

                model="phi3",

                messages=[

                    {
                        "role": "system",
                        "content": "You are a helpful AI Research Paper Assistant. Answer in simple English."
                    },

                    {
                        "role": "user",
                        "content": f"""
Research Paper:

{best_chunk}

Question:

{question}

Answer in simple English.
"""
                    }

                ]
            )

            st.subheader("Answer")
            st.write(
                response["message"]["content"]
            )
            


        st.divider()

        if st.button("📝 Generate Summary"):

            with st.spinner("Generating summary..."):

                summary = ollama.chat(

                    model="phi3",

                    messages=[

                        {
                            "role": "system",
                            "content": "Summarize research papers in simple English."
                        },

                        {
                            "role": "user",
                            "content": f"""
Summarize this research paper.

{text[:12000]}
"""
                        }

                    ]
                )

            st.subheader("Research Paper Summary")
            st.write(summary["message"]["content"])


# ==================================================
# KEY POINTS
# ==================================================

        st.divider()

        if st.button("📌 Extract Key Points"):

            with st.spinner("Extracting key points..."):

                keypoints = ollama.chat(

                    model="phi3",

                    messages=[

                        {
                            "role": "system",
                            "content": "Extract the important points from research papers."
                        },

                        {
                            "role": "user",
                            "content": f"""
Read the research paper and provide:

* Research Objective

* Problem Statement

* Methodology

* Dataset

* Findings

* Results

* Limitations

* Conclusion

Research Paper:

{text[:12000]}
"""
                        }

                    ]
                )

            st.subheader("Key Points")
            st.write(keypoints["message"]["content"])


# ==================================================
# RESUME ANALYZER
# ==================================================

elif feature == "📋 Resume Analyzer":

    st.header("📋 AI Resume Analyzer")

    resume_file = st.file_uploader(
        "Upload Resume (PDF)",
        type="pdf"
    )

    job_role = st.text_input(
        "Enter Target Job Role"
    )

    if resume_file is not None and st.button("Analyze Resume"):

        resume_text = extract_pdf_text(resume_file)

        with st.spinner("Analyzing Resume..."):

            analysis = ollama.chat(

                model="phi3",

                messages=[

                    {
                        "role": "system",
                        "content": """
You are an expert Resume Analyzer.

Analyze the resume and provide:

1. Skills Present

2. Missing Skills

3. Missing Projects

4. Missing Achievements

5. Resume Weaknesses

6. Resume Score out of 100

7. Recommendations to improve the resume.

Use simple English.
"""
                    },

                    {
                        "role": "user",
                        "content": f"""
Target Job Role:

{job_role}

Resume:

{resume_text[:12000]}
"""
                    }

                ]
            )

        st.subheader("Resume Analysis")

        st.write(
            analysis["message"]["content"]
        )
