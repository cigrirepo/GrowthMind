import os
import json
from openai import OpenAI
import streamlit as st

# Instantiate the new client once
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def call_openai(prompt: str, model="gpt-4o-mini") -> str:
    if not client.api_key:
        st.error("🔑 OPENAI_API_KEY not set!")
        st.stop()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=1000,
    )
    return resp.choices[0].message.content.strip()

# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="SELF-DISCOVER Assistant", layout="wide")
st.title("🧠 SELF-DISCOVER Reasoning Assistant")

task = st.text_area("Describe the complex task you need to solve:")

if st.button("Run SELF-DISCOVER"):
    if not task:
        st.error("Please enter a task above.")
    else:
        # 1. Build the 3-stage prompt
        prompt = f"""
You are a strategic reasoning assistant applying the SELF-DISCOVER framework to complex tasks. 
Your role is not just to answer, but to reason first.
## Instructions: Apply the following 3-stage reasoning process to the task:
### 1. SELECT 
From a list of reasoning strategies below, select the most relevant modules for solving the problem.
Strategies include: Break down into sub-tasks; Evaluate unit economics; Use systems thinking; Analyse competitor positioning; Identify pricing inefficiencies; Spot underused growth loops; Model conversion bottlenecks; Explore localisation opportunities; Prioritise by impact × feasibility; Conduct dynamic SWOT analysis
### 2. ADAPT 
Rephrase each selected module into a task-specific step, tailored to the problem at hand.
### 3. IMPLEMENT 
Output your plan as a structured JSON object, where each key is a reasoning step and each value is a plain-language description of what you will do.
---
## Stage 2 – Execute the Reasoning Plan:
Use your own plan to solve the task:
- Generate actionable insights  
- Identify 3–5 untapped strategic levers  
- Score each lever on **Impact** and **Feasibility** (scale of 1–5)  
- Recommend top 2–3 priorities based on overall score  
- Clearly note assumptions, unknowns, or data gaps
---
## Task:
{task}
## Output Format:
```json
{{
  "selected_modules": [...],
  "adapted_structure": {{ "Step 1": "...", … }},
  "opportunity_gaps": ["1. …", …],
  "prioritized_actions": [{{ "action": "...", "impact": X, "feasibility": Y, … }}]
}}
```
"""
        # 2. Call OpenAI
        with st.spinner("🤖 Thinking…"):
            try:
                raw = call_openai(prompt)
                # 3. Parse & display
                # Clean the response by removing markdown code block formatting
                cleaned_response = raw
                # Remove markdown code blocks (```json and ```)
                if cleaned_response.startswith("```json"):
                    cleaned_response = cleaned_response.replace("```json", "", 1)
                if cleaned_response.endswith("```"):
                    cleaned_response = cleaned_response[:-3]
                
                # Strip any leading/trailing whitespace
                cleaned_response = cleaned_response.strip()
                
                try:
                    data = json.loads(cleaned_response)
                    st.json(data)
                except json.JSONDecodeError:
                    st.warning("⚠️ Response wasn't valid JSON — showing raw output:")
                    st.code(raw)
            except Exception as e:
                st.error(f"Error calling OpenAI API: {str(e)}")
