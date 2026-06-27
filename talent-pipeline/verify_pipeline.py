# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0

import asyncio
import os
from app.pipeline import run_talent_pipeline
from dotenv import load_dotenv

# Load local environment variables
load_dotenv()

async def main():
    print("--------------------------------------------------")
    print("🚀 Starting End-to-End Pipeline Verification")
    print("--------------------------------------------------")
    
    # Check API Key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY not found in environment!")
        return
    else:
        print(f"🔑 GEMINI_API_KEY found: {api_key[:5]}...{api_key[-5:]}")
        
    print(f"📍 GOOGLE_GENAI_USE_VERTEXAI = {os.environ.get('GOOGLE_GENAI_USE_VERTEXAI')}")
    
    # Short sample job description for quick testing
    sample_job = """
    We are looking for a Senior AI Systems Engineer.
    Core Challenges:
    1. Optimize inference throughput for open LLMs.
    2. Set up multi-GPU distributed training infrastructure.
    
    Must-haves: PyTorch, vLLM, Kubernetes.
    """
    
    # Custom logger for progress
    def progress_log(message: str):
        print(f"[Pipeline] {message}")
        
    try:
        results = await run_talent_pipeline(
            job_description=sample_job,
            top_k=3,
            progress_callback=progress_log
        )
        
        job_profile = results["job_profile"]
        ranked_list = results["ranked_shortlist"]
        
        print("\n==================================================")
        print("🎉 PIPELINE RUN SUCCESSFUL!")
        print("==================================================")
        print(f"Parsed Job Title: {job_profile.title}")
        print(f"Core Challenges Extracted: {job_profile.core_technical_challenges}")
        print(f"Must-Haves: {job_profile.must_haves}")
        print("\n--- Ranked Shortlist ---")
        
        for idx, item in enumerate(ranked_list):
            cand = item["candidate"]
            evaluation = item["evaluation"]
            score = item["hybrid_score"]
            print(f"\n#{idx+1}: {cand.name} (Hybrid Score: {score:.2f}/10.0)")
            print(f"  - Technical Fit: {evaluation.technical_fit}/10.0")
            print(f"  - Culture & Trajectory Fit: {evaluation.culture_trajectory_fit}/10.0")
            print(f"  - Delivery Capability: {evaluation.delivery_capability}/10.0")
            print(f"  - Recruiter Reasoning: {evaluation.justification[:120]}...")
            
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
