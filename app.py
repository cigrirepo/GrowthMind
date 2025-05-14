import os
import json
import re
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
            {"role": "system", "content": "You are a strategic business consultant with expertise in growth strategies."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=1500,
    )
    return resp.choices[0].message.content.strip()

def generate_implementation_plan(data, selected_action):
    prompt = f"""
    Create a detailed 30-60-90 day implementation plan for this strategic action:
    
    {selected_action}
    
    The plan should include:
    1. Specific milestones for each time period (30, 60, and 90 days)
    2. Key metrics to track success
    3. Required resources and estimated costs
    4. Potential challenges and mitigation strategies
    
    Return as a structured JSON object with these exact keys:
    - "thirty_day_plan": [list of objects with "milestone", "details", "metrics" keys]
    - "sixty_day_plan": [list of objects with "milestone", "details", "metrics" keys]
    - "ninety_day_plan": [list of objects with "milestone", "details", "metrics" keys]
    - "success_metrics": [list of objects with "metric", "target", "tracking_method" keys]
    - "resources_required": [list of objects with "resource", "purpose", "estimated_cost" keys]
    - "potential_challenges": [list of objects with "challenge", "mitigation_strategy" keys]
    
    Return only valid JSON without any code block markers or additional text.
    """
    
    response = call_openai(prompt)
    
    # Clean the response
    cleaned_response = response
    if cleaned_response.startswith("```json"):
        cleaned_response = cleaned_response.replace("```json", "", 1)
    if cleaned_response.endswith("```"):
        cleaned_response = cleaned_response[:-3]
    cleaned_response = cleaned_response.strip()
    
    try:
        implementation_data = json.loads(cleaned_response)
        return implementation_data
    except json.JSONDecodeError:
        st.warning("⚠️ Implementation plan response wasn't valid JSON.")
        return None

def calculate_roi_projection(action_data):
    prompt = f"""
    Create a 12-month ROI projection for this strategic action:
    
    {action_data}
    
    Return only a JSON array of 12 monthly values representing estimated percentage increase in revenue.
    Example: [1.5, 2.3, 3.1, 4.0, 5.2, 5.8, 6.5, 7.1, 7.5, 8.0, 8.3, 8.5]
    Do not include any code block markers, explanation text, or anything other than the array itself.
    """
    
    response = call_openai(prompt)
    
    # Extract array from response using regex
    match = re.search(r'\[[\d\., ]+\]', response)
    if match:
        array_str = match.group(0)
        try:
            roi_data = json.loads(array_str)
            return roi_data
        except json.JSONDecodeError:
            return [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]  # Default fallback
    else:
        return [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]  # Default fallback

def generate_competitive_analysis(industry, opportunity):
    prompt = f"""
    Conduct a brief competitive analysis related to this opportunity in the {industry} industry:
    
    {opportunity}
    
    Focus on:
    1. What leading competitors are doing in this area
    2. Market benchmarks and best practices
    3. Potential competitive advantages
    
    Return as JSON with these exact keys:
    - "competitor_approaches": [list of objects with "competitor", "approach", "effectiveness" keys]
    - "market_benchmarks": [list of benchmark strings]
    - "competitive_advantage_opportunities": [list of opportunity strings]
    
    Return only valid JSON without any code block markers or additional text.
    """
    
    response = call_openai(prompt)
    
    # Clean the response
    cleaned_response = response
    if cleaned_response.startswith("```json"):
        cleaned_response = cleaned_response.replace("```json", "", 1)
    if cleaned_response.endswith("```"):
        cleaned_response = cleaned_response[:-3]
    cleaned_response = cleaned_response.strip()
    
    try:
        analysis_data = json.loads(cleaned_response)
        return analysis_data
    except json.JSONDecodeError:
        st.warning("⚠️ Competitive analysis response wasn't valid JSON.")
        return None

# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="SELF-DISCOVER Growth Strategy Consultant", layout="wide")

# Session state initialization
if 'results' not in st.session_state:
    st.session_state.results = None
if 'implementation_plan' not in st.session_state:
    st.session_state.implementation_plan = None
if 'competitive_analysis' not in st.session_state:
    st.session_state.competitive_analysis = None
if 'selected_action' not in st.session_state:
    st.session_state.selected_action = None
if 'roi_data' not in st.session_state:
    st.session_state.roi_data = None

# Header and intro
st.title("🚀 SELF-DISCOVER Growth Strategy Consultant")
st.markdown("""
This AI-powered consultant helps you identify strategic growth opportunities and provides 
actionable implementation plans using the SELF-DISCOVER framework.
""")

# Sidebar for inputs
with st.sidebar:
    st.header("Business Context")
    company_name = st.text_input("Company Name", "")
    industry = st.selectbox("Industry", 
                          ["Restaurant", "Retail", "Technology", "Healthcare", "Education", 
                           "Manufacturing", "Finance", "Professional Services", "Other"])
    
    company_size = st.radio("Company Size", 
                          ["Startup (<10 employees)", "Small (10-50 employees)", 
                           "Medium (51-250 employees)", "Large (>250 employees)"])
    
    model_choice = st.selectbox("AI Model", 
                              ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"], 
                              help="More advanced models may provide better results but can be slower")

# Main content area
task_col1, task_col2 = st.columns([2, 1])

with task_col1:
    task = st.text_area("Describe the complex business challenge or growth opportunity:", 
                      height=150,
                      placeholder="Example: We run a local restaurant in Portland, Maine, and need to increase revenue by 20% next year while maintaining our current quality standards...")

with task_col2:
    st.markdown("### Focus Areas (Optional)")
    focus_areas = st.multiselect(
        "Select specific areas you want to prioritize:",
        ["Customer Acquisition", "Customer Retention", "Pricing Strategy", 
         "Product Development", "Market Expansion", "Cost Optimization",
         "Digital Transformation", "Marketing Effectiveness"]
    )

# Run analysis button
run_col1, run_col2 = st.columns([1, 3])
with run_col1:
    run_button = st.button("🔍 Run SELF-DISCOVER Analysis", type="primary", use_container_width=True)

if run_button:
    if not task:
        st.error("Please enter a business challenge above.")
    else:
        # 1. Build the enhanced 3-stage prompt
        focus_areas_text = ""
        if focus_areas:
            focus_areas_text = "Focus especially on these priority areas: " + ", ".join(focus_areas)
            
        company_context = f"Company: {company_name} (Industry: {industry}, Size: {company_size})"
        
        prompt = f"""
You are a strategic growth consultant applying the SELF-DISCOVER framework to complex business challenges.
Your role is to deliver actionable, high-value strategic insights.

## Context
{company_context}

## Instructions: Apply the following 3-stage reasoning process:
### 1. SELECT 
From this list of strategic growth modules, select the most relevant ones for solving this business challenge:
- Break down into sub-tasks
- Evaluate unit economics
- Use systems thinking
- Analyse competitor positioning
- Identify pricing inefficiencies
- Spot underused growth loops
- Model conversion bottlenecks
- Explore localisation opportunities
- Prioritise by impact × feasibility
- Conduct dynamic SWOT analysis

### 2. ADAPT 
Rephrase each selected module into a specific step for this business challenge.

### 3. IMPLEMENT 
Analyze the challenge using your adapted framework to:
- Generate 5 specific, actionable strategic opportunities
- Score each on Impact (1-5) and Feasibility (1-5)
- Prioritize based on combined score

{focus_areas_text}

## Challenge:
{task}

## Output Format:
```json
{{
  "selected_modules": ["List of selected strategy modules"],
  "adapted_structure": {{ "Step 1": "Description", "Step 2": "Description", ... }},
  "opportunity_gaps": ["1. First specific opportunity", "2. Second opportunity", ...],
  "prioritized_actions": [
    {{ "action": "First priority action", "impact": X, "feasibility": Y }},
    {{ "action": "Second priority action", "impact": X, "feasibility": Y }},
    ...
  ]
}}
```
Return ONLY valid JSON without any additional text or explanations.
"""
        # 2. Call OpenAI with enhanced context
        with st.spinner("🤖 Analyzing your business challenge..."):
            raw = call_openai(prompt, model=model_choice)
        
        # 3. Clean and parse results
        cleaned_response = raw
        # Remove markdown code blocks
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response.replace("```json", "", 1)
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
        cleaned_response = cleaned_response.strip()
        
        try:
            data = json.loads(cleaned_response)
            st.session_state.results = data
        except json.JSONDecodeError:
            st.warning("⚠️ Response wasn't valid JSON — showing raw output:")
            st.code(raw)
            st.session_state.results = None

# Display results if available
if st.session_state.results:
    data = st.session_state.results
    
    # Display key insights in tabs
    tabs = st.tabs(["Strategic Framework", "Opportunity Analysis", "Implementation Planning"])
    
    with tabs[0]:
        st.header("Strategic Growth Framework")
        
        # Selected modules and adapted structure
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Selected Strategy Modules")
            for i, module in enumerate(data["selected_modules"]):
                st.markdown(f"**{i+1}.** {module}")
        
        with col2:
            st.subheader("Adapted Growth Framework")
            for step, description in data["adapted_structure"].items():
                st.markdown(f"**{step}:** {description}")
    
    with tabs[1]:
        st.header("Strategic Growth Opportunities")
        
        # Opportunity gaps
        st.subheader("Identified Opportunity Gaps")
        for opportunity in data["opportunity_gaps"]:
            st.markdown(f"• {opportunity}")
        
        # Prioritized actions
        st.subheader("Prioritized Strategic Actions")
        
        # Create a DataFrame for the prioritized actions
        actions_data = []
        for action in data["prioritized_actions"]:
            actions_data.append({
                "Action": action["action"],
                "Impact": action["impact"],
                "Feasibility": action["feasibility"],
                "Total Score": action["impact"] + action["feasibility"]
            })
        
        # Display prioritized actions in a table format
        for i, action in enumerate(data["prioritized_actions"]):
            impact = action["impact"]
            feasibility = action["feasibility"]
            total = impact + feasibility
            
            st.markdown(f"""
            **Priority {i+1}: {action['action']}**
            - Impact: {'⭐' * impact} ({impact}/5)
            - Feasibility: {'🔄' * feasibility} ({feasibility}/5)
            - Total Score: {total}/10
            """)
        
        # Action selection for detailed planning
        st.subheader("Select an Action for Detailed Planning")
        action_options = [action["action"] for action in data["prioritized_actions"]]
        selected_action = st.selectbox("Choose a strategic action to develop:", action_options)
        
        if selected_action != st.session_state.selected_action:
            st.session_state.selected_action = selected_action
            st.session_state.implementation_plan = None
            st.session_state.competitive_analysis = None
            st.session_state.roi_data = None
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Generate Implementation Plan", use_container_width=True):
                with st.spinner("Creating detailed implementation plan..."):
                    st.session_state.implementation_plan = generate_implementation_plan(
                        data, selected_action
                    )
                    st.session_state.roi_data = calculate_roi_projection(selected_action)
                    
        with col2:
            if st.button("Generate Competitive Analysis", use_container_width=True):
                with st.spinner("Analyzing competitive landscape..."):
                    st.session_state.competitive_analysis = generate_competitive_analysis(
                        industry, selected_action
                    )
        
    with tabs[2]:
        st.header("Detailed Implementation Planning")
        
        if st.session_state.implementation_plan:
            plan = st.session_state.implementation_plan
            
            # Display 30-60-90 day plan in columns
            st.subheader("30-60-90 Day Implementation Plan")
            plan_col1, plan_col2, plan_col3 = st.columns(3)
            
            with plan_col1:
                st.markdown("### 30-Day Milestones")
                for milestone in plan["thirty_day_plan"]:
                    with st.expander(milestone["milestone"]):
                        st.markdown(f"**Details:** {milestone['details']}")
                        st.markdown(f"**Metrics:** {milestone['metrics']}")
            
            with plan_col2:
                st.markdown("### 60-Day Milestones")
                for milestone in plan["sixty_day_plan"]:
                    with st.expander(milestone["milestone"]):
                        st.markdown(f"**Details:** {milestone['details']}")
                        st.markdown(f"**Metrics:** {milestone['metrics']}")
            
            with plan_col3:
                st.markdown("### 90-Day Milestones")
                for milestone in plan["ninety_day_plan"]:
                    with st.expander(milestone["milestone"]):
                        st.markdown(f"**Details:** {milestone['details']}")
                        st.markdown(f"**Metrics:** {milestone['metrics']}")
            
            # Success metrics
            st.subheader("Key Success Metrics")
            for metric in plan["success_metrics"]:
                st.markdown(f"""
                **{metric['metric']}**
                - Target: {metric['target']}
                - Tracking Method: {metric['tracking_method']}
                """)
            
            # Resources required
            st.subheader("Resources Required")
            for resource in plan["resources_required"]:
                st.markdown(f"""
                **{resource['resource']}**
                - Purpose: {resource['purpose']}
                - Estimated Cost: {resource['estimated_cost']}
                """)
            
            # Challenges and mitigation
            st.subheader("Risk Management")
            for challenge in plan["potential_challenges"]:
                with st.expander(f"Challenge: {challenge['challenge']}"):
                    st.markdown(f"**Mitigation Strategy:** {challenge['mitigation_strategy']}")
            
            # ROI Projection
            if st.session_state.roi_data:
                st.subheader("12-Month ROI Projection")
                months = ["Month " + str(i+1) for i in range(12)]
                
                # Display ROI data in a readable format
                st.markdown("### Projected Monthly Revenue Increase (%)")
                roi_text = ""
                for i, val in enumerate(st.session_state.roi_data):
                    roi_text += f"**Month {i+1}:** +{val}%\n\n"
                st.markdown(roi_text)
                
                # Calculate cumulative effect
                cumulative = sum(st.session_state.roi_data)
                st.success(f"**Estimated 12-Month Revenue Impact:** +{cumulative:.1f}%")
        
        if st.session_state.competitive_analysis:
            analysis = st.session_state.competitive_analysis
            
            st.subheader("Competitive Landscape Analysis")
            
            # Competitor approaches
            st.markdown("### Competitor Approaches")
            for comp in analysis["competitor_approaches"]:
                st.markdown(f"""
                **{comp['competitor']}**
                - Approach: {comp['approach']}
                - Effectiveness: {comp['effectiveness']}
                """)
            
            # Market benchmarks
            st.markdown("### Market Benchmarks")
            for benchmark in analysis["market_benchmarks"]:
                st.markdown(f"• {benchmark}")
            
            # Competitive advantages
            st.markdown("### Competitive Advantage Opportunities")
            for advantage in analysis["competitive_advantage_opportunities"]:
                st.markdown(f"• {advantage}")
        
        if not st.session_state.implementation_plan and not st.session_state.competitive_analysis:
            st.info("Select an action from the Opportunity Analysis tab and generate an implementation plan or competitive analysis to see detailed planning information.")

# Footer with export option
st.markdown("---")
st.markdown("""
### Next Steps

1. **Review the strategic framework** to understand the reasoning behind the recommendations
2. **Explore each opportunity** in detail to assess its potential impact
3. **Generate implementation plans** for your highest priority actions
4. **Export or share** your strategic growth plan with stakeholders

If you need to make changes, adjust your business challenge description or focus areas and run the analysis again.
""")
