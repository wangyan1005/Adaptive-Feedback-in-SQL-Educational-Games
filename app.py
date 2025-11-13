import streamlit as st
import numpy as np
import pickle
import faiss
import pandas as pd
from openai import OpenAI
from adaptive_feedback_pipeline_v2 import generate_sql_feedback  
import json, re


client = OpenAI()

# Load FAISS + example metadata
@st.cache_resource
def load_retrieval_store():
    index = faiss.read_index("db/embeddings.faiss")
    metadata = pickle.load(open("db/example_meta.pkl", "rb"))
    return index, metadata

index, metadata = load_retrieval_store()

def pick_random_user_profile(path="sql_engagement_dataset.csv"):
    """Load dataset from path and randomly pick one user."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()

    row = df.sample(1).iloc[0]

    profile = {
        "typing_speed": float(row["typing_speed"]),
        "avg_flight_time": float(row["avg_flight_time"]),
        "avg_dwell_time": float(row["avg_dwell_time"]),
        "backspace_rate": float(row["backspace_rate"]),
        "delete_rate": float(row["delete_rate"]),
        "retry_count": int(row["retry_count"]),
        "emotion": row["emotion"],
        "learner_type": row["learner_type"]
    }

    return profile
  
# Streamlit app configuration
st.set_page_config(page_title="Adaptive SQL Feedback System", layout="wide")

# Inject custom CSS
st.markdown(
    """
    <style>
        /* Center the title */
        .title {
            text-align: center;
            color: #2C3E50;
        }

        /* Move the text area slightly down */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        h1 {
            margin-bottom: 40px !important;
        }
        
        .element-container:has(.stSubheader) {
            margin-top: -10px !important;
        }
        
        div.stButton {
            text-align: center;
        }
        
        div.stButton > button {
            display: inline-block !important;
            margin: 0.5rem auto !important;
            background-color: black !important;
            color: white !important;
            border: none !important;
            border-radius: 6px !important;
            padding: 0.5rem 1.5rem !important;
            font-size: 16px !important;
            font-weight: 500 !important;
            cursor: pointer !important;
            transition: all 0.3s ease !important;
        }
        
        div.stButton > button:hover {
            background-color: #333333 !important;
            color: white !important;
        }
        
        div.stButton > button:active {
            background-color: #555555 !important;
        }

        /* Text area styling */
        textarea {
            border-radius: 6px !important;
            padding: 10px !important;
        }

        textarea:focus {
            outline: none !important;
            border: 1.5px solid black !important;
            box-shadow: none !important;
        }
        
        [data-testid="column"]:first-child > div {
            gap: 0.5rem !important;
        }
        
        .stTextArea {
            margin-bottom: 0 !important;
        }

    </style>
    """,
    unsafe_allow_html=True
)

# Centered Title 
st.markdown(
    """
    <h1 style='text-align: center; margin-bottom: 20px; color: #2C3E50;'>
        Adaptive SQL Learning Feedback System
    </h1>
    """,
    unsafe_allow_html=True
)

# Create layout
col_left, col_right = st.columns([1, 1.2], gap="large")

# Left Panel: Query Input
with col_left:
    st.subheader("Enter Your SQL Query:")
    sql_query = st.text_area(
        label="",
        placeholder="e.g., SELECT Employee_ID, Name FROM Employees WHERE Employee_ID = 'John';",
        height=130
    )

    # Submit button 
    submitted = st.button("Submit Query")

# Right Panel: Display results only after submit
if submitted:
    if not sql_query.strip():
        with col_right:
            st.warning("Please enter a SQL query first.")
    else:
        user_profile = pick_random_user_profile()
       
        def extract_json(text: str):
            # Remove Markdown code fences
            text = re.sub(r"```json|```", "", text).strip()
            # Find JSON object inside the text
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                raise ValueError("No JSON object found in the LLM response.")

        # Generate feedback using LLM
        try:
            llm_raw = generate_sql_feedback(
                query=sql_query,
                user_profile=user_profile,
            )
            
            # Extract structured JSON from LLM output
            llm_output = extract_json(llm_raw)
            
            error_type_predicted = llm_output["error_type"]
            error_subtype_predicted = llm_output["error_subtype"]
            final_feedback = llm_output["personalized_feedback"]
            
        except Exception as e:
            with col_right:
                st.error(f"Error processing query: {str(e)}")
                st.error(f"Raw LLM output: {llm_raw}")  
            st.stop()


        # Display results
        with col_right:
            st.subheader("Results")
            st.markdown(f"""
                <p style='font-size: 20px; margin: 10px 0;'>
                    <strong>Error Type:</strong> <code style='font-size: 20px; background-color: #f0f2f6; padding: 4px 8px; border-radius: 4px;'>{error_type_predicted}</code>
                </p>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
                <p style='font-size: 20px; margin: 10px 0;'>
                    <strong>Error Subtype:</strong> <code style='font-size: 20px; background-color: #f0f2f6; padding: 4px 8px; border-radius: 4px;'>{error_subtype_predicted}</code>
                </p>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
                <p style='font-size: 20px; margin: 10px 0;'>
                    <strong>Learner Type:</strong> <code style='font-size: 20px; background-color: #f0f2f6; padding: 4px 8px; border-radius: 4px;'>{user_profile["learner_type"].replace("_", " ")}</code>
                </p>
            """, unsafe_allow_html=True)

            st.divider()
            st.subheader("Feedback")
            st.markdown(f"""
                <div style='background-color: #d1ecf1; border: 1px solid #bee5eb; border-radius: 8px; padding: 15px; margin-top: 10px;'>
                    <p style='font-size: 20px; line-height: 1.6; margin: 0; color: #0c5460;'>
                        {final_feedback}
                    </p>
                </div>
            """, unsafe_allow_html=True)

else:
    with col_right:
        st.empty()

