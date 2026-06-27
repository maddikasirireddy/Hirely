# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0

import asyncio
import json
from typing import List, Dict, Any, Callable, Optional
from app.agents import (
    job_understanding_agent,
    recruiter_reasoning_agent,
    JobProfile,
    CandidateProfile,
    CandidateEvaluation
)
from app.vector_store import CandidateVectorStore
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# Singleton instance of VectorStore so it's loaded only once across the application lifetime
_vector_store_instance = None

async def get_vector_store(progress_callback: Optional[Callable[[str], None]] = None) -> CandidateVectorStore:
    """Gets or initializes the global Candidate Vector Store."""
    global _vector_store_instance
    if _vector_store_instance is None:
        store = CandidateVectorStore()
        await store.initialize(progress_callback)
        _vector_store_instance = store
    return _vector_store_instance


# =====================================================================
# Pipeline Orchestration
# =====================================================================

async def run_talent_pipeline(
    job_description: str,
    top_k: int = 5,
    progress_callback: Optional[Callable[[str], None]] = None
) -> Dict[str, Any]:
    """Runs the complete talent acquisition pipeline.
    
    1. Runs JobUnderstandingAgent to extract True Intent.
    2. Runs FAISS Vector Search to retrieve top candidates.
    3. Runs RecruiterReasoningAgent on each candidate.
    4. Combines vector search scores with LLM evaluation for hybrid ranking.
    5. Returns sorted candidates and the structured job profile.
    """
    if progress_callback:
        progress_callback("🎬 Starting the Smart Talent Acquisition Pipeline...")

    session_service = InMemorySessionService()
    
    # -----------------------------------------------------------------
    # Step 1: Analyze Job Description
    # -----------------------------------------------------------------
    if progress_callback:
        progress_callback("🤖 Step 1: Running Job Understanding Agent to extract 'True Intent'...")
        
    job_session_id = "job_analysis_session"
    await session_service.create_session(app_name="app", user_id="recruiter", session_id=job_session_id)
    
    job_runner = Runner(
        agent=job_understanding_agent,
        app_name="app",
        session_service=session_service
    )
    
    async for event in job_runner.run_async(
        user_id="recruiter",
        session_id=job_session_id,
        new_message=types.Content(role="user", parts=[types.Part.from_text(text=job_description)])
    ):
        pass
        
    job_session = await session_service.get_session(app_name="app", user_id="recruiter", session_id=job_session_id)
    job_profile_raw = job_session.state.get("job_profile")
    
    if not job_profile_raw:
        raise ValueError("Failed to analyze the Job Description. Please check your inputs and API configuration.")
        
    if isinstance(job_profile_raw, dict):
        job_profile = JobProfile(**job_profile_raw)
    else:
        job_profile = job_profile_raw
        
    if progress_callback:
        progress_callback(f"✅ Job Analyzed: '{job_profile.title}'. Extracted {len(job_profile.core_technical_challenges)} core challenges.")

    # -----------------------------------------------------------------
    # Step 2: FAISS Semantic Retrieval
    # -----------------------------------------------------------------
    if progress_callback:
        progress_callback("🔍 Step 2: Initializing FAISS Vector Store and retrieving top cohort...")
        
    store = await get_vector_store(progress_callback)
    
    if progress_callback:
        progress_callback(f"🔎 Querying FAISS index for the top {top_k} matching candidates...")
        
    retrieved_cohort = store.retrieve(job_profile, top_k=top_k)
    
    if progress_callback:
        progress_callback(f"✅ Retrieved {len(retrieved_cohort)} candidates from vector store.")

    # -----------------------------------------------------------------
    # Step 3: Recruiter Reasoning Evaluation
    # -----------------------------------------------------------------
    if progress_callback:
        progress_callback("🧠 Step 3: Running Recruiter Reasoning Agent on retrieved candidates...")
        
    evaluated_candidates = []
    
    for idx, (candidate, faiss_score) in enumerate(retrieved_cohort):
        if progress_callback:
            progress_callback(f"💬 Evaluating Candidate {idx+1}/{len(retrieved_cohort)}: {candidate.name} (FAISS Similarity: {faiss_score:.2f})...")
            
        # Serialize profiles to JSON for the LLM prompt
        # Using custom serializer helpers to avoid Pydantic serialization nuances
        from app.vector_store import serialize_profile
        job_profile_dict = job_profile.model_dump()
        candidate_profile_dict = serialize_profile(candidate)
        
        evaluation_prompt = (
            f"Please evaluate this candidate against the job requirements.\n\n"
            f"--- JOB PROFILE ---\n"
            f"{json.dumps(job_profile_dict, indent=2)}\n\n"
            f"--- CANDIDATE PROFILE ---\n"
            f"{json.dumps(candidate_profile_dict, indent=2)}"
        )
        
        reasoning_session_id = f"eval_session_{candidate.name.lower().replace(' ', '_')}"
        await session_service.create_session(app_name="app", user_id="recruiter", session_id=reasoning_session_id)
        
        reasoning_runner = Runner(
            agent=recruiter_reasoning_agent,
            app_name="app",
            session_service=session_service
        )
        
        async for _ in reasoning_runner.run_async(
            user_id="recruiter",
            session_id=reasoning_session_id,
            new_message=types.Content(role="user", parts=[types.Part.from_text(text=evaluation_prompt)])
        ):
            pass
            
        reasoning_session = await session_service.get_session(app_name="app", user_id="recruiter", session_id=reasoning_session_id)
        evaluation_raw = reasoning_session.state.get("candidate_evaluation")
        
        if not evaluation_raw:
            # Fallback evaluation in case of API issues
            evaluation = CandidateEvaluation(
                candidate_name=candidate.name,
                technical_fit=5.0,
                culture_trajectory_fit=5.0,
                delivery_capability=5.0,
                justification="Recruiter Reasoning Agent failed to generate evaluation. Displaying fallback scores."
            )
        else:
            if isinstance(evaluation_raw, dict):
                evaluation = CandidateEvaluation(**evaluation_raw)
            else:
                evaluation = evaluation_raw
            
        # -------------------------------------------------------------
        # Step 4: Hybrid Scoring Algorithm
        # -------------------------------------------------------------
        # FAISS Score is in [0, 1], LLM metrics are in [0, 10]
        # We scale FAISS Score to [0, 10]
        faiss_scaled = faiss_score * 10.0
        llm_average = (evaluation.technical_fit + evaluation.culture_trajectory_fit + evaluation.delivery_capability) / 3.0
        
        # Weighted hybrid score: 30% FAISS (semantic embedding match), 70% LLM reasoning (deeper analysis)
        hybrid_score = (0.3 * faiss_scaled) + (0.7 * llm_average)
        
        evaluated_candidates.append({
            "candidate": candidate,
            "evaluation": evaluation,
            "faiss_score": faiss_score,
            "hybrid_score": hybrid_score
        })

    # -----------------------------------------------------------------
    # Step 5: Consolidate & Sort Final Shortlist
    # -----------------------------------------------------------------
    if progress_callback:
        progress_callback("📊 Step 4: Consolidating scores and sorting the final shortlist...")
        
    # Sort candidates by hybrid_score in descending order
    ranked_shortlist = sorted(evaluated_candidates, key=lambda x: x["hybrid_score"], reverse=True)
    
    if progress_callback:
        progress_callback("🏆 Pipeline completed successfully! Shortlist generated.")
        
    return {
        "job_profile": job_profile,
        "ranked_shortlist": ranked_shortlist
    }
