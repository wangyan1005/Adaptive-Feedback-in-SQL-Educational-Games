import faiss
import pickle
import numpy as np
from openai import OpenAI

client = OpenAI()

EMBED_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4o"  

# Load FAISS + metadata
index = faiss.read_index("db/embeddings.faiss")
metadata = pickle.load(open("db/example_meta.pkl", "rb"))

def embed_query(query: str):
    emb = client.embeddings.create(
        model=EMBED_MODEL,
        input=query
    ).data[0].embedding
    return np.array(emb).astype("float32")


def retrieve_similar_examples(query: str, k=3):
    q_vec = embed_query(query)
    q_vec = q_vec.reshape(1, -1)

    distances, indices = index.search(q_vec, k)
    return [metadata[i] for i in indices[0]]


def build_prompt(query, examples, user_profile):
    SCHEMA = """
    ### Database Schema
        Employees(
            Employee_ID INT PK, 
            Name TEXT, 
            Job_Role TEXT, 
            Division TEXT, 
            Last_Login_Time DATETIME
        )

        Robots(
            Robot_ID INT PK, 
            Model TEXT, 
            Manufacturing_Date DATETIME, 
            Status TEXT, 
            Last_Software_Update DATETIME, 
            Employee_ID INT FK
        )

        Logs(
            Log_ID INT PK, 
            Employee_ID INT FK, 
            Action_Description TEXT, 
            Timestamp DATETIME, 
            Robot_ID INT FK
        )

        Incidents(
            Incident_ID INT PK, 
            Description TEXT, 
            Timestamp DATETIME, 
            Robot_ID INT FK, 
            Employee_ID INT FK
        )

        Access_Codes(
            Access_Code_ID INT PK, 
            Employee_ID INT FK, 
            Level_of_Access TEXT, 
            Timestamp_of_Last_Use DATETIME  
        )
    """
    
    TAXONOMY = """
    Error Types:
    1. Syntax Error
        - misspelling
        - missing quotes
        - missing commas
        - missing semicolons
        - non-standard operators
        - unmatched brackets
        - data type mismatch
        - incorrect wildcard usage
        - incomplete query
        - incorrect SELECT usage
        - incorrect DISTINCT usage
        - wrong positioning
        - aggregation misuse
    
    2. Schema Error
        - undefined table
        - undefined column
        - undefined function

    3. Logic Error
        - ambiguous reference
        - incorrect GROUP BY usage
        - incorrect HAVING clause
        - incorrect JOIN usage
        - incorrect ORDER BY usage
        - operator misuse
    
    4. Construction Error
        - inefficient query
    """

    USER_BEHAVIOR_RULES = """
    ### User Behavior Interpretation Rules 

    Typing Speed:
    - < 2.1 keys/s → slow and careful 
    - 2.1–3.4 keys/s → normal pace 
    - > 3.4 keys/s → fast and energetic 

    Dwell Time:
    - < 90 ms → quick decisive keypresses
    - 90–122 ms → normal dwell time 
    - > 122 ms → thoughtful, cautious pressing 

    Flight Time:
    - < 195 ms → very fast transitions 
    - 195–380 ms → normal transitions 
    - > 380 ms → longer pauses, possible uncertainty

    Correction Rate (backspace/delete combined):
    - < 0.02 → very low correction behavior
    - 0.02–0.08 → normal corrections
    - > 0.08 → high correction behavior
    """

    SQL_SEMICOLON_RULE = """
    ### Semicolon Rules
    - Single SQL statements typically don't require semicolons in most environments
    - Mark Error type as "Syntax" and Error subtype as "missing semicolons" if it ends without a semicolon
    """

    shots = "\n\n".join([
        f"### Example {i+1}\n"
        f"SQL Query: {ex['query']}\n"
        f"Error Type: {ex['error_type']}\n"
        f"Error Subtype: {ex['error_subtype']}\n"
        f"Feedback: {ex['feedback']}\n"
        for i, ex in enumerate(examples)
    ])

    user_text = f"""
### User Profile
Typing Speed: {user_profile['typing_speed']} keys/sec
Flight Time: {user_profile['avg_flight_time']} ms
Dwell Time: {user_profile['avg_dwell_time']} ms
Backspace Rate: {user_profile['backspace_rate']}
Delete Rate: {user_profile['delete_rate']}
Retry Count: {user_profile['retry_count']}
Emotion: {user_profile['emotion']}
"""

    return f"""
You are an intelligent SQL debugging tutor.

{SCHEMA}
{TAXONOMY}
{USER_BEHAVIOR_RULES}
{SQL_SEMICOLON_RULE}
{user_text}

### Task
Given the new SQL query, analyze it using the style shown in the examples.
Return **strict JSON** in the following format:

{{
  "error_type": "",
  "error_subtype": "",
  "personalized_feedback": ""
}}

Rules for personalized_feedback:
- Structure: [Behavior observation] + [Emotion acknowledgment] + [Technical hint] + [Emotional support]
- Tone adaptation based on emotion AND behavior:
  * angry + fast typing → "I see you're working quickly through this challenge. [hint]. Take a breath, you've got this!"
  * sad + slow typing → "It's okay to take your time. [hint]. Every step forward counts!"
  * happy + normal typing → "Great energy! [hint]. Keep up the momentum!"
  * neutral + any → Focus on behavior and hint
  
- Behavior observation examples:
  * Fast typing: "Your quick pace shows you're eager to solve this"
  * Slow typing: "Taking a thoughtful approach is smart"
  * High corrections: "I notice you're refining your work carefully"
  * Low corrections: "Your confident keystrokes show good focus"

- Emotion-specific encouragement:
  * angry → Calming: "Take a breath", "One step at a time", "You're closer than you think"
  * sad → Uplifting: "You're doing better than you realize", "Every attempt is progress", "This is how learning happens"
  * happy → Reinforcing: "Great energy!", "Keep that momentum!", "You're on the right track"
  * neutral → Standard: "Keep going!", "You've got this!", "Almost there!"

- Technical hint: Must be specific to the error but not reveal the answer

- Length: 2-3 sentences total

### Few-Shot Examples
{shots}

### New SQL Query
{query}

### JSON Response:
"""

def generate_sql_feedback(query, user_profile):
    examples = retrieve_similar_examples(query, k=2)
    prompt = build_prompt(query, examples, user_profile)

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )

    content = response.choices[0].message.content
    return content

    