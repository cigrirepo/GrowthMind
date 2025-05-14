import os
import json
import re
import pandas as pd
import numpy as np
import streamlit as st
from openai import OpenAI
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from io import StringIO

# Instantiate the OpenAI client once
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Set up caching for expensive operations
@st.cache_data(ttl=3600)
def call_openai(prompt: str, model="gpt-4o-mini") -> str:
    """Call OpenAI API with caching to reduce redundant API calls"""
    if not client.api_key:
        st.error("🔑 OPENAI_API_KEY not set!")
        st.stop()
    
    try:
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
    except Exception as e:
        st.error(f"Error calling OpenAI API: {str(e)}")
        return None

def parse_json_response(response):
    """Clean and parse JSON responses more robustly"""
    # Clean the response of any markdown code blocks and whitespace
    cleaned_response = response.strip()
    if cleaned_response.startswith("```json"):
        cleaned_response = cleaned_response.replace("```json", "", 1)
    if cleaned_response.endswith("```"):
        cleaned_response = cleaned_response[:-3]
    cleaned_response = cleaned_response.strip()
    
    # Try to parse the JSON response
    try:
        return json.loads(cleaned_response)
    except json.JSONDecodeError:
        # If parsing fails, try a more aggressive approach to extract JSON
        try:
            # Find content that looks like JSON (between curly braces)
            match = re.search(r'({.*})', cleaned_response, re.DOTALL)
            if match:
                return json.loads(match.group(1))
        except:
            # If all parsing attempts fail, return None
            return None

def generate_implementation_plan(data, selected_action):
    """Generate a detailed implementation plan for a selected strategic action"""
    prompt = f"""
    Create a detailed 30-60-90 day implementation plan for this strategic action:
    
    {selected_action}
    
    The plan should include:
    1. Specific milestones for each time period (30, 60, and 90 days)
    2. Key metrics to track success
    3. Required resources and estimated costs
    4. Potential challenges and mitigation strategies
    5. Implementation timeline with dependencies
    6. Clear roles and responsibilities for stakeholders
    
    Return as a structured JSON object with these exact keys:
    - "thirty_day_plan": [list of objects with "milestone", "details", "metrics", "owner" keys]
    - "sixty_day_plan": [list of objects with "milestone", "details", "metrics", "owner" keys]
    - "ninety_day_plan": [list of objects with "milestone", "details", "metrics", "owner" keys]
    - "success_metrics": [list of objects with "metric", "target", "tracking_method", "reporting_frequency" keys]
    - "resources_required": [list of objects with "resource", "purpose", "estimated_cost", "acquisition_timeline" keys]
    - "potential_challenges": [list of objects with "challenge", "impact_level", "probability", "mitigation_strategy" keys]
    - "stakeholder_roles": [list of objects with "role", "responsibilities", "deliverables" keys]
    
    Return only valid JSON without any code block markers or additional text.
    """
    
    response = call_openai(prompt)
    if not response:
        return None
    
    return parse_json_response(response)

def calculate_roi_projection(action_data, company_size, industry):
    """Calculate ROI projection based on action, company size, and industry"""
    prompt = f"""
    Create a detailed 12-month ROI projection for this strategic action in a {company_size} {industry} company:
    
    {action_data}
    
    Return a JSON object with these exact keys:
    - "monthly_revenue_increases": [12 monthly values representing estimated percentage increase in revenue]
    - "monthly_cost_projections": [12 monthly values representing estimated costs in thousands of dollars]
    - "breakeven_point": [month number when ROI becomes positive]
    - "cumulative_roi_percentage": [cumulative ROI percentage after 12 months]
    - "key_assumptions": [list of assumption strings that were made in this projection]
    
    Do not include any code block markers, explanation text, or anything other than the JSON object itself.
    """
    
    response = call_openai(prompt)
    if not response:
        return None
    
    return parse_json_response(response)

def generate_competitive_analysis(industry, opportunity):
    """Generate competitive analysis with enhanced detail"""
    prompt = f"""
    Conduct a comprehensive competitive analysis related to this opportunity in the {industry} industry:
    
    {opportunity}
    
    Focus on:
    1. What leading competitors are doing in this area
    2. Market benchmarks and best practices
    3. Potential competitive advantages
    4. Market size and growth projections
    5. Barriers to entry and critical success factors
    6. SWOT analysis for the top 3 competitors
    
    Return as JSON with these exact keys:
    - "competitor_approaches": [list of objects with "competitor", "approach", "effectiveness", "market_share" keys]
    - "market_benchmarks": [list of objects with "benchmark", "industry_average", "best_in_class" keys]
    - "competitive_advantage_opportunities": [list of objects with "opportunity", "difficulty", "potential_impact" keys]
    - "market_analysis": { "size": "market size in $B", "cagr": "compound annual growth rate", "trends": [list of trend strings] }
    - "barriers_to_entry": [list of barrier strings]
    - "critical_success_factors": [list of factor strings]
    - "competitor_swot": [list of objects with "competitor", "strengths", "weaknesses", "opportunities", "threats" keys]
    
    Return only valid JSON without any code block markers or additional text.
    """
    
    response = call_openai(prompt)
    if not response:
        return None
    
    return parse_json_response(response)

def generate_risk_assessment(action, industry):
    """Generate detailed risk assessment for a strategic action"""
    prompt = f"""
    Create a comprehensive risk assessment for implementing this strategic action in the {industry} industry:
    
    {action}
    
    Include:
    1. Identification of key risks (financial, operational, market, regulatory, etc.)
    2. Assessment of probability and impact for each risk
    3. Recommended mitigation strategies
    4. Contingency plans for high-impact risks
    
    Return as JSON with these exact keys:
    - "risks": [list of objects with "category", "description", "probability", "impact", "risk_score" keys]
    - "mitigation_strategies": [list of objects with "risk_id", "strategy", "resource_requirements", "timeline" keys]
    - "contingency_plans": [list of objects with "risk_id", "trigger_conditions", "response_plan", "responsible_party" keys]
    - "risk_matrix": { "high_priority": [list of risk IDs], "medium_priority": [list of risk IDs], "low_priority": [list of risk IDs] }
    
    Return only valid JSON without any code block markers or additional text.
    """
    
    response = call_openai(prompt)
    if not response:
        return None
    
    return parse_json_response(response)

def generate_financial_projections(action, company_size, industry):
    """Generate detailed financial projections for a strategic action"""
    prompt = f"""
    Create detailed financial projections for implementing this strategic action in a {company_size} {industry} company:
    
    {action}
    
    Include:
    1. Initial investment requirements
    2. Monthly cash flow projections for 12 months
    3. P&L impact summary
    4. Key financial metrics (ROI, IRR, Payback Period)
    5. Sensitivity analysis with best/worst case scenarios
    
    Return as JSON with these exact keys:
    - "initial_investment": { "capex": value, "opex": value, "total": value }
    - "monthly_projections": [list of 12 objects with "month", "revenue", "expenses", "cash_flow" keys]
    - "pl_impact": { "year_1_revenue_impact": value, "year_1_profit_impact": value, "margin_impact": value }
    - "financial_metrics": { "roi_percentage": value, "irr_percentage": value, "payback_period_months": value, "npv": value }
    - "sensitivity_analysis": { "best_case": {metrics}, "expected_case": {metrics}, "worst_case": {metrics} }
    
    Return only valid JSON without any code block markers or additional text.
    """
    
    response = call_openai(prompt)
    if not response:
        return None
    
    return parse_json_response(response)

def generate_kpi_dashboard(action, industry):
    """Generate KPI dashboard structure for measuring success"""
    prompt = f"""
    Create a comprehensive KPI dashboard structure for tracking the success of this strategic action in the {industry} industry:
    
    {action}
    
    Include:
    1. Leading indicators (early warning signs of success/failure)
    2. Lagging indicators (outcome measures)
    3. Operational metrics
    4. Financial metrics
    5. Customer metrics
    
    Return as JSON with these exact keys:
    - "leading_indicators": [list of objects with "metric", "target", "measurement_frequency", "data_source" keys]
    - "lagging_indicators": [list of objects with "metric", "target", "measurement_frequency", "data_source" keys]
    - "operational_metrics": [list of objects with "metric", "target", "measurement_frequency", "data_source" keys]
    - "financial_metrics": [list of objects with "metric", "target", "measurement_frequency", "data_source" keys]
    - "customer_metrics": [list of objects with "metric", "target", "measurement_frequency", "data_source" keys]
    - "dashboard_sections": [list of dashboard section names]
    
    Return only valid JSON without any code block markers or additional text.
    """
    
    response = call_openai(prompt)
    if not response:
        return None
    
    return parse_json_response(response)

def plot_roi_projection(roi_data):
    """Create an interactive ROI projection chart using Plotly"""
    if not roi_data or "monthly_revenue_increases" not in roi_data:
        return None
    
    months = [f"Month {i+1}" for i in range(12)]
    revenue_increases = roi_data["monthly_revenue_increases"]
    
    # Calculate cumulative increases
    cumulative = []
    current_sum = 0
    for val in revenue_increases:
        current_sum += val
        cumulative.append(current_sum)
    
    # Create a DataFrame for plotting
    df = pd.DataFrame({
        "Month": months,
        "Monthly Increase (%)": revenue_increases,
        "Cumulative Increase (%)": cumulative
    })
    
    # Create a Plotly figure
    fig = go.Figure()
    
    # Add bar chart for monthly increases
    fig.add_trace(go.Bar(
        x=df["Month"],
        y=df["Monthly Increase (%)"],
        name="Monthly Revenue Increase (%)",
        marker_color="rgb(55, 83, 109)"
    ))
    
    # Add line chart for cumulative increases
    fig.add_trace(go.Scatter(
        x=df["Month"],
        y=df["Cumulative Increase (%)"],
        name="Cumulative Revenue Increase (%)",
        marker_color="rgb(26, 118, 255)",
        mode="lines+markers"
    ))
    
    # Update layout
    fig.update_layout(
        title="12-Month ROI Projection",
        xaxis_title="Month",
        yaxis_title="Revenue Increase (%)",
        legend_title="Metric",
        template="plotly_white"
    )
    
    return fig

def plot_action_comparison(actions_data):
    """Create an interactive comparison chart of prioritized actions"""
    if not actions_data:
        return None
    
    # Create a DataFrame for plotting
    df = pd.DataFrame(actions_data)
    
    # Sort by total score
    df = df.sort_values("Total Score", ascending=False)
    
    # Create a Plotly figure
    fig = go.Figure()
    
    # Add bar charts for impact and feasibility
    fig.add_trace(go.Bar(
        x=df["Action"],
        y=df["Impact"],
        name="Impact",
        marker_color="rgb(26, 118, 255)"
    ))
    
    fig.add_trace(go.Bar(
        x=df["Action"],
        y=df["Feasibility"],
        name="Feasibility",
        marker_color="rgb(55, 83, 109)"
    ))
    
    # Update layout
    fig.update_layout(
        title="Strategic Actions: Impact vs. Feasibility",
        xaxis_title="Action",
        yaxis_title="Score (1-5)",
        legend_title="Metric",
        template="plotly_white",
        barmode="group"
    )
    
    return fig

def plot_risk_matrix(risk_data):
    """Create a risk matrix visualization"""
    if not risk_data or "risks" not in risk_data:
        return None
    
    # Create a DataFrame for risks
    risks = pd.DataFrame(risk_data["risks"])
    
    # Create a Plotly figure
    fig = go.Figure()
    
    # Add scatter plot for risks
    fig.add_trace(go.Scatter(
        x=risks["probability"],
        y=risks["impact"],
        mode="markers+text",
        marker=dict(
            size=12,
            color=risks["risk_score"],
            colorscale="RdYlGn_r",
            showscale=True,
            colorbar=dict(title="Risk Score")
        ),
        text=risks["description"],
        textposition="top center",
        name="Risks"
    ))
    
    # Add risk zones
    fig.add_shape(
        type="rect",
        x0=0, y0=3.5, x1=5, y1=5,
        line=dict(color="rgba(255,0,0,0.1)", width=0),
        fillcolor="rgba(255,0,0,0.1)"
    )
    
    fig.add_shape(
        type="rect",
        x0=3.5, y0=0, x1=5, y1=3.5,
        line=dict(color="rgba(255,0,0,0.1)", width=0),
        fillcolor="rgba(255,0,0,0.1)"
    )
    
    fig.add_shape(
        type="rect",
        x0=0, y0=0, x1=1.5, y1=1.5,
        line=dict(color="rgba(0,255,0,0.1)", width=0),
        fillcolor="rgba(0,255,0,0.1)"
    )
    
    # Update layout
    fig.update_layout(
        title="Risk Assessment Matrix",
        xaxis_title="Probability",
        yaxis_title="Impact",
        xaxis=dict(range=[0, 5], tickvals=[1, 2, 3, 4, 5]),
        yaxis=dict(range=[0, 5], tickvals=[1, 2, 3, 4, 5]),
        template="plotly_white"
    )
    
    fig.add_annotation(
        x=4.5, y=4.5,
        text="High Risk",
        showarrow=False,
        font=dict(color="red")
    )
    
    fig.add_annotation(
        x=0.75, y=0.75,
        text="Low Risk",
        showarrow=False,
        font=dict(color="green")
    )
    
    return fig

def export_to_pdf(data, filename="strategy_plan.pdf"):
    """Export strategy plan to PDF format"""
    # This would require additional libraries like ReportLab
    # Placeholder for now
    return "PDF export functionality would be implemented here"

def create_exec_summary(data, company_name, industry, action):
    """Generate an executive summary of the strategy plan"""
    prompt = f"""
    Create a concise executive summary for the following strategic action plan for {company_name} in the {industry} industry:
    
    {action}
    
    Summary of analysis results:
    {json.dumps(data)}
    
    The executive summary should include:
    1. Background and context
    2. Key findings and recommendations
    3. Expected outcomes and benefits
    4. Implementation approach
    5. Critical success factors
    
    Keep the summary under 500 words and focus on business impact and value.
    """
    
    response = call_openai(prompt)
    if not response:
        return "Unable to generate executive summary."
    
    return response
