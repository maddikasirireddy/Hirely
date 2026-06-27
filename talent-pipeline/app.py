# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0

import streamlit as st
import asyncio
import os
import textwrap
import pandas as pd
from dotenv import load_dotenv

# Load local env variables
load_dotenv()

# Set Streamlit page config
st.set_page_config(
    page_title="Hirely",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# Theme CSS definitions
# =====================================================================

LIGHT_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    .stApp { background-color: #f5f7fa; color: #1e293b; font-family: 'Inter', sans-serif; }
    .stApp p, .stApp li, .stApp label, .stApp span { color: #1e293b !important; }
    h1, h2, h3, h4, h5, h6 { color: #0f172a !important; }
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e2e8f0; }
    .gradient-text {
        background: linear-gradient(135deg, #4f46e5 0%, #6366f1 50%, #818cf8 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-weight: 800; font-size: 2.8rem; margin-bottom: 0.2rem;
    }
    .subtitle { color: #475569; font-size: 1.1rem; margin-bottom: 2rem; }
    .candidate-card {
        background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px;
        padding: 1.5rem; margin-bottom: 1.5rem;
        box-shadow: 0 2px 12px rgba(99,102,241,0.07);
        transition: transform 0.2s, border-color 0.2s, box-shadow 0.2s;
    }
    .candidate-card:hover { border-color: #6366f1; transform: translateY(-2px); box-shadow: 0 6px 24px rgba(99,102,241,0.15); }
    .score-circle {
        background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%);
        border-radius: 50%; width: 80px; height: 80px;
        display: flex; align-items: center; justify-content: center;
        border: 2px solid #c7d2fe; box-shadow: 0 0 18px rgba(99,102,241,0.3); flex-shrink: 0;
    }
    .score-val { font-size: 1.6rem; font-weight: 800; color: #ffffff; }
    .metric-box { background-color: #f1f5f9; border-radius: 10px; padding: 0.8rem; text-align: center; border: 1px solid #e2e8f0; }
    .metric-title { font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600; }
    .metric-value { font-size: 1.25rem; font-weight: 700; color: #1e293b; margin-top: 0.25rem; }
    .recruiter-quote {
        border-left: 4px solid #6366f1; background-color: #eef2ff;
        padding: 1rem 1.25rem; margin: 1rem 0; border-radius: 0 10px 10px 0;
        font-style: italic; color: #374151; line-height: 1.6;
    }
    .badge-expert { background-color: #d1fae5; color: #065f46; padding: 0.2rem 0.6rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; }
    .badge-intermediate { background-color: #fef3c7; color: #92400e; padding: 0.2rem 0.6rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; }
    .badge-novice { background-color: #f1f5f9; color: #475569; padding: 0.2rem 0.6rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; }
    div[data-baseweb="textarea"] { background-color: #ffffff !important; border-color: #cbd5e1 !important; }
    textarea { color: #1e293b !important; background-color: #ffffff !important; }
    .status-container { background-color: #f8fafc; border: 1px dashed #6366f1; border-radius: 8px; padding: 1rem; margin-bottom: 1.5rem; color: #374151; }
    [data-testid="stTab"] { color: #475569 !important; font-weight: 500; }
    [data-testid="stTab"][aria-selected="true"] { color: #4f46e5 !important; border-bottom-color: #4f46e5 !important; }
    [data-baseweb="select"] { background-color: #ffffff !important; }
    [data-baseweb="select"] * { color: #1e293b !important; }
    .stButton > button { background: linear-gradient(135deg, #4f46e5, #6366f1) !important; color: #ffffff !important; border: none !important; font-weight: 600 !important; border-radius: 8px !important; }
    .stButton > button:hover { background: linear-gradient(135deg, #4338ca, #4f46e5) !important; box-shadow: 0 4px 16px rgba(79,70,229,0.35) !important; }
    .theme-toggle-btn { position: fixed; top: 14px; right: 60px; z-index: 9999; }
    header[data-testid="stHeader"] { background-color: transparent !important; }
    div[data-testid="stDecoration"] { background-color: transparent !important; }
</style>
"""

DARK_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    .stApp { background-color: #0d0f12; color: #e2e8f0; font-family: 'Inter', sans-serif; }
    .stApp p, .stApp li, .stApp label, .stApp span { color: #e2e8f0 !important; }
    h1, h2, h3, h4, h5, h6 { color: #f1f5f9 !important; }
    [data-testid="stSidebar"] { background-color: #10131a; border-right: 1px solid #1e2536; }
    .gradient-text {
        background: linear-gradient(135deg, #a5b4fc 0%, #6366f1 50%, #4f46e5 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-weight: 800; font-size: 2.8rem; margin-bottom: 0.2rem;
    }
    .subtitle { color: #94a3b8; font-size: 1.1rem; margin-bottom: 2rem; }
    .candidate-card {
        background-color: #151922; border: 1px solid #242b3d; border-radius: 14px;
        padding: 1.5rem; margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.35);
        transition: transform 0.2s, border-color 0.2s, box-shadow 0.2s;
    }
    .candidate-card:hover { border-color: #4f46e5; transform: translateY(-2px); box-shadow: 0 6px 30px rgba(99,102,241,0.2); }
    .score-circle {
        background: linear-gradient(135deg, #312e81 0%, #4338ca 100%);
        border-radius: 50%; width: 80px; height: 80px;
        display: flex; align-items: center; justify-content: center;
        border: 2px solid #6366f1; box-shadow: 0 0 18px rgba(99,102,241,0.45); flex-shrink: 0;
    }
    .score-val { font-size: 1.6rem; font-weight: 800; color: #ffffff; }
    .metric-box { background-color: #1e2530; border-radius: 10px; padding: 0.8rem; text-align: center; border: 1px solid #2d3748; }
    .metric-title { font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600; }
    .metric-value { font-size: 1.25rem; font-weight: 700; color: #f1f5f9; margin-top: 0.25rem; }
    .recruiter-quote {
        border-left: 4px solid #6366f1; background-color: #1b1e27;
        padding: 1rem 1.25rem; margin: 1rem 0; border-radius: 0 10px 10px 0;
        font-style: italic; color: #cbd5e1; line-height: 1.6;
    }
    .badge-expert { background-color: #065f46; color: #34d399; padding: 0.2rem 0.6rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; }
    .badge-intermediate { background-color: #78350f; color: #fbbf24; padding: 0.2rem 0.6rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; }
    .badge-novice { background-color: #1e293b; color: #94a3b8; padding: 0.2rem 0.6rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; }
    div[data-baseweb="textarea"] { background-color: #151922 !important; border-color: #242b3d !important; }
    textarea { color: #e2e8f0 !important; background-color: #151922 !important; }
    .status-container { background-color: #11141b; border: 1px dashed #4338ca; border-radius: 8px; padding: 1rem; margin-bottom: 1.5rem; color: #94a3b8; }
    [data-testid="stTab"] { color: #94a3b8 !important; font-weight: 500; }
    [data-testid="stTab"][aria-selected="true"] { color: #818cf8 !important; border-bottom-color: #818cf8 !important; }
    [data-baseweb="select"], [data-baseweb="select"] > div { background-color: #151922 !important; border-color: #242b3d !important; }
    [data-baseweb="select"] * { color: #e2e8f0 !important; }
    ul[role="listbox"], ul[role="listbox"] li, ul[role="listbox"] div { background-color: #151922 !important; color: #e2e8f0 !important; }
    .stSelectbox div[data-baseweb="select"] span { color: #e2e8f0 !important; }
    .stButton > button { background: linear-gradient(135deg, #4f46e5, #6366f1) !important; color: #ffffff !important; border: none !important; font-weight: 600 !important; border-radius: 8px !important; }
    .stButton > button:hover { background: linear-gradient(135deg, #4338ca, #4f46e5) !important; box-shadow: 0 4px 16px rgba(99,102,241,0.4) !important; }
    .theme-toggle-btn { position: fixed; top: 14px; right: 60px; z-index: 9999; }
    header[data-testid="stHeader"] { background-color: transparent !important; }
    div[data-testid="stDecoration"] { background-color: transparent !important; }
</style>
"""


# Imports from our modular pipeline
from app.pipeline import run_talent_pipeline, get_vector_store
from app.vector_store import MOCK_RESUMES
from app.agents import JobProfile

# =====================================================================
# 1. Preset Job Templates
# =====================================================================

PRESET_JOBS = {
    "Senior AI/ML Platform Engineer": """
NexusAI is seeking a Senior ML Platform Engineer to build and scale our model training and inference pipelines.
Core Challenges to Solve:
1. Optimize inference latency for open-source LLMs (Llama 3, Mistral) processing over 10M tokens daily.
2. Architect a scalable multi-GPU distributed training platform on Kubernetes.
3. Build robust model monitoring and GPU memory management tools to minimize cold-start latencies.

Must-Haves:
- Strong PyTorch experience in multi-node, distributed environments (DDP, FSDP).
- Deep understanding of inference engines like vLLM, TensorRT-LLM, or Triton Inference Server.
- Hands-on experience with Kubernetes, Docker, and GPU orchestration.

Nice-to-Haves:
- Contributions to PyTorch core or related open-source ML systems.
- Experience writing custom CUDA kernels.
    """.strip(),

    "Staff Backend & Distributed Systems Engineer": """
CloudScale Solutions is looking for a Staff Software Engineer to lead the architecture of our high-throughput events mesh.
Core Challenges to Solve:
1. Re-architect our core transactional backbone to handle a 5x increase in concurrent users (over 10M transactions daily).
2. Design and implement a low-latency custom transactional append-only log storage engine.
3. Establish robust engineering patterns, testing frameworks, and reduce systemic production incidents.

Must-Haves:
- Expert-level proficiency in Go or Rust.
- Deep expertise in concurrent programming, distributed consensus, and database internals (e.g. PostgreSQL, Redis, Kafka).
- Proven track record of architecting high-scale distributed systems in production.

Nice-to-Haves:
- Open-source database or distributed systems libraries.
- Active speaker or contributor in the systems engineering community.
    """.strip(),

    "Principal Frontend Architect": """
Designify is hiring a Principal Frontend Architect to own our web application architecture and design system.
Core Challenges to Solve:
1. Migrate a massive, monolithic React codebase into a highly performant micro-frontend architecture using Webpack Module Federation or Vite.
2. Build and scale a highly accessible (WAI-ARIA compliant), reusable component library.
3. Implement advanced, highly performant 3D data visualization features.

Must-Haves:
- Expert-level React, TypeScript, and modern bundlers (Webpack, Vite).
- Deep understanding of web performance profiling, asset optimizations, and render trees.
- Exceptional expertise in accessibility standards (a11y, WCAG, WAI-ARIA).

Nice-to-Haves:
- Hands-on experience with Three.js, WebGL, or Canvas APIs.
- Core contributor to open-source UI/component frameworks.
    """.strip()
}

# Initialize session states
if "pipeline_results" not in st.session_state:
    st.session_state.pipeline_results = None
if "job_description_input" not in st.session_state:
    st.session_state.job_description_input = PRESET_JOBS["Senior AI/ML Platform Engineer"]
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Dashboard"
if "theme" not in st.session_state:
    st.session_state.theme = "light"

# Inject active theme CSS
if st.session_state.theme == "dark":
    st.markdown(DARK_CSS, unsafe_allow_html=True)
else:
    st.markdown(LIGHT_CSS, unsafe_allow_html=True)

# =====================================================================
# 2. Main Interface Layout
# =====================================================================

# Header row: title + theme toggle
_hcol_title, _hcol_spacer, _hcol_toggle = st.columns([6, 1, 1])
with _hcol_title:
    st.markdown('<div class="gradient-text">Hirely ⚡</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Deep Semantic Candidate Alignment & Recruiter Reasoning Pipeline</div>', unsafe_allow_html=True)
with _hcol_toggle:
    _toggle_label = "☀️ Light" if st.session_state.theme == "dark" else "🌙 Dark"
    if st.button(_toggle_label, key="theme_toggle"):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.rerun()

# Tabs
tab_dash, tab_pool, tab_intent = st.tabs(["🎯 Match Dashboard", "👥 Candidate Pool & Custom Resumes", "🧠 Job True Intent Analysis"])

# =====================================================================
# TAB 1: DASHBOARD
# =====================================================================
with tab_dash:
    col_input, col_results = st.columns([1, 1.3])
    
    with col_input:
        st.subheader("Job Specification")
        
        # Template Selector
        selected_template = st.selectbox(
            "Select a Job Template to Auto-fill:",
            ["Custom (Paste your own)"] + list(PRESET_JOBS.keys()),
            index=1  # Default to AI/ML Engineer
        )
        
        # Handle Template Selection
        if selected_template != "Custom (Paste your own)":
            st.session_state.job_description_input = PRESET_JOBS[selected_template]
            
        # Raw Job Description input
        job_desc = st.text_area(
            "Paste Job Description:",
            value=st.session_state.job_description_input,
            height=320,
            key="job_desc_field"
        )
        
        # Config options
        cohort_size = st.slider("FAISS Retrieval Cohort Size (Evaluated by LLM):", min_value=3, max_value=9, value=5)
        
        # Run button
        st.write("")
        if st.button("🚀 Run Multi-Agent Matching Pipeline", use_container_width=True):
            if not job_desc.strip():
                st.error("Please enter a valid job description.")
            else:
                # Execution container
                status_box = st.empty()
                log_entries = []
                
                def update_log(message: str):
                    log_entries.append(message)
                    with status_box.container():
                        st.markdown('<div class="status-container">', unsafe_allow_html=True)
                        st.markdown("**🔄 Agent Execution Logs:**")
                        for entry in log_entries:
                            st.write(entry)
                        st.markdown('</div>', unsafe_allow_html=True)
                
                try:
                    # Run the async pipeline synchronously in Streamlit
                    with st.spinner("Processing..."):
                        results = asyncio.run(run_talent_pipeline(
                            job_desc,
                            top_k=cohort_size,
                            progress_callback=update_log
                        ))
                    
                    st.session_state.pipeline_results = results
                    st.success("🎉 Talent Matching Pipeline completed successfully!")
                except Exception as e:
                    st.error(f"Error executing pipeline: {str(e)}")
                    st.info("Ensure your GEMINI_API_KEY is active and valid.")
                    
    with col_results:
        st.subheader("Ranked Shortlist")
        
        if st.session_state.pipeline_results is None:
            st.info("👈 Enter a Job Description and click 'Run' to generate candidate rankings and deep reasoning.")
        else:
            results = st.session_state.pipeline_results
            ranked_shortlist = results["ranked_shortlist"]
            
            for idx, item in enumerate(ranked_shortlist):
                cand = item["candidate"]
                eval_data = item["evaluation"]
                faiss_score = item["faiss_score"]
                hybrid_score = item["hybrid_score"]
                
                # Render Candidate Card (theme-aware inline colors)
                _name_color = "#0f172a" if st.session_state.theme == "light" else "#f1f5f9"
                _meta_color = "#64748b" if st.session_state.theme == "light" else "#94a3b8"
                st.markdown(textwrap.dedent(f"""
                <div class="candidate-card">
                    <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 1rem;">
                        <div>
                            <span style="font-size: 1.4rem; font-weight: 700; color: {_name_color};">#{idx+1} {cand.name}</span>
                            <div style="color: {_meta_color}; font-size: 0.9rem; margin-top: 0.2rem;">
                                FAISS Similarity: {faiss_score:.2f} | Hybrid Index: {hybrid_score:.2f}/10.0
                            </div>
                        </div>
                        <div class="score-circle">
                            <div class="score-val">{hybrid_score:.1f}</div>
                        </div>
                    </div>

                    <!-- Score breakdown grid -->
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 1rem;">
                        <div class="metric-box">
                            <div class="metric-title">Technical Fit</div>
                            <div class="metric-value">{eval_data.technical_fit:.1f}/10</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-title">Culture & Trajectory</div>
                            <div class="metric-value">{eval_data.culture_trajectory_fit:.1f}/10</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-title">Delivery Capability</div>
                            <div class="metric-value">{eval_data.delivery_capability:.1f}/10</div>
                        </div>
                    </div>

                    <!-- Recruiter Reasoning Quote -->
                    <div class="recruiter-quote">
                        <strong>💬 Recruiter Reasoning:</strong><br>
                        {eval_data.justification}
                    </div>
                </div>
                """), unsafe_allow_html=True)

                
                # Expandable details
                with st.expander(f"View Deep Semantic Profile for {cand.name}"):
                    st.markdown(f"**Career Trajectory Summary:**\n{cand.semantic_trajectory}")
                    st.write("")
                    
                    st.markdown("**Core Skills Depth Assessment:**")
                    cols = st.columns(len(cand.skills_depth))
                    for i, skill in enumerate(cand.skills_depth):
                        with cols[i % len(cols)]:
                            badge_cls = "badge-expert" if skill.level == "Expert" else ("badge-intermediate" if skill.level == "Intermediate" else "badge-novice")
                            st.markdown(f"**{skill.skill}** <span class='{badge_cls}'>{skill.level}</span>", unsafe_allow_html=True)
                            st.caption(skill.context_justification)
                            
                    st.write("")
                    st.markdown(f"**Behavioral & Leadership Signals:**")
                    for signal in cand.behavioral_signals:
                        st.markdown(f"- {signal}")
                        
                    st.write("")
                    st.markdown(f"**Open Source & Platform Activity:**")
                    for act in cand.platform_activity:
                        st.markdown(f"- {act}")

# =====================================================================
# TAB 2: CANDIDATE POOL & CUSTOM RESUMES
# =====================================================================
with tab_pool:
    st.subheader("Manage Talent Database & Index")
    
    col_list, col_add = st.columns([1.2, 1])
    
    with col_list:
        st.markdown("### Pre-Seeded Candidates")
        st.markdown("This local pool represents your existing talent network. Run the pipeline on the main tab to search against them.")
        
        # Load active store to show current candidates
        try:
            store = asyncio.run(get_vector_store())
            active_profiles = store.profiles
        except Exception:
            active_profiles = []
            
        if not active_profiles:
            st.info("Loading candidates... Click 'Run' on the Dashboard or initialize the store.")
            # Small button to force initialize
            if st.button("Initialize Candidate Database"):
                with st.spinner("Initializing FAISS Store..."):
                    store = asyncio.run(get_vector_store(lambda x: st.write(x)))
                st.rerun()
        else:
            # Track which candidate is pending delete confirmation
            if "pending_delete" not in st.session_state:
                st.session_state.pending_delete = None

            for idx, cand in enumerate(active_profiles):
                with st.expander(f"👤 {cand.name} — parsed profile"):
                    st.markdown(f"**Semantic Trajectory:**\n{cand.semantic_trajectory}")
                    st.markdown("**Skills Depth:**")
                    for s in cand.skills_depth:
                        badge_color = "🟢" if s.level == "Expert" else ("🟡" if s.level == "Intermediate" else "⚪")
                        st.markdown(f"- {badge_color} **{s.skill}** ({s.level}): *{s.context_justification}*")
                    st.markdown("**Behavioral Signals:** " + ", ".join(cand.behavioral_signals))
                    st.markdown("**Platform Activity:** " + ", ".join(cand.platform_activity))

                    st.write("")
                    st.divider()

                    if st.session_state.pending_delete == cand.name:
                        # Confirmation step
                        st.warning(f"⚠️ Are you sure you want to remove **{cand.name}** from the talent pool? This will also update the FAISS index.")
                        col_confirm, col_cancel = st.columns(2)
                        with col_confirm:
                            if st.button(f"✅ Confirm Remove", key=f"confirm_{idx}"):
                                with st.spinner(f"Removing {cand.name} and rebuilding index..."):
                                    removed = store.remove_candidate(cand.name)
                                if removed:
                                    # Reset the singleton so next pipeline run uses updated store
                                    import app.pipeline as _pipeline_mod
                                    _pipeline_mod._vector_store_instance = store
                                    st.session_state.pending_delete = None
                                    st.success(f"✅ {cand.name} removed from the talent pool.")
                                    st.rerun()
                                else:
                                    st.error(f"Could not find {cand.name} in the index.")
                        with col_cancel:
                            if st.button("❌ Cancel", key=f"cancel_{idx}"):
                                st.session_state.pending_delete = None
                                st.rerun()
                    else:
                        if st.button(f"🗑️ Remove {cand.name}", key=f"delete_{idx}"):
                            st.session_state.pending_delete = cand.name
                            st.rerun()


    with col_add:
        st.markdown("### Add Custom Candidate (Dynamic Parsing & Indexing)")
        st.markdown("Paste a raw resume here. The **Candidate Understanding Agent** will parse it, extract semantic traits, and dynamically insert it into the **FAISS index** for instant query capability!")
        
        new_name = st.text_input("Candidate Full Name:", placeholder="e.g. John Doe")
        new_resume = st.text_area("Paste Raw Resume Text:", height=250, placeholder="Work History, Skills, Achievements...")
        
        if st.button("➕ Parse and Index Candidate"):
            if not new_name.strip() or not new_resume.strip():
                st.error("Please enter both Name and Resume text.")
            else:
                try:
                    with st.spinner(f"Parsing and embedding {new_name}..."):
                        store = asyncio.run(get_vector_store())
                        profile = asyncio.run(store.add_candidate(new_name, new_resume))
                    st.success(f"🎉 Successfully parsed {new_name}! Added to local FAISS index.")
                    st.balloons()
                    
                    # Display what was parsed
                    st.markdown("#### Agent Extracted Profile:")
                    st.markdown(f"**Trajectory:** {profile.semantic_trajectory}")
                    st.markdown("**Skills Extracted:**")
                    for s in profile.skills_depth:
                        st.markdown(f"- **{s.skill}** ({s.level}): {s.context_justification}")
                except Exception as e:
                    st.error(f"Error parsing candidate: {str(e)}")

# =====================================================================
# TAB 3: TRUE INTENT ANALYSIS
# =====================================================================
with tab_intent:
    st.subheader("Job Specification Deep Analysis")
    
    if st.session_state.pipeline_results is None:
        st.info("Please run the pipeline on the 'Dashboard' tab first to generate the Job True Intent Profile.")
    else:
        results = st.session_state.pipeline_results
        job_profile: JobProfile = results["job_profile"]
        
        st.markdown(f"### Role: {job_profile.title}")
        
        col_challenges, col_expectations = st.columns(2)
        
        with col_challenges:
            st.markdown("#### 🛠️ Core Technical Challenges to Solve")
            st.markdown("These represent the real, complex engineering problems the candidate must solve, extracted by the agent:")
            for ch in job_profile.core_technical_challenges:
                st.markdown(f"- {ch}")
                
            st.markdown("#### 📋 Requirements Differentiated")
            col_must, col_nice = st.columns(2)
            with col_must:
                st.markdown("**Must-Haves (Dealbreakers):**")
                for item in job_profile.must_haves:
                    st.markdown(f"- {item}")
            with col_nice:
                st.markdown("**Nice-to-Haves (Value Adds):**")
                for item in job_profile.nice_to_haves:
                    st.markdown(f"- {item}")
                    
        with col_expectations:
            st.markdown("#### 📈 Ideal Career Trajectory")
            st.write(job_profile.required_trajectory)
            
            st.markdown("#### 🌟 Implicitly Expected Traits & Mindsets")
            st.markdown("Skills and attributes that recruiters expect but are rarely written in the job description:")
            for trait in job_profile.implicitly_expected_traits:
                st.markdown(f"- {trait}")
