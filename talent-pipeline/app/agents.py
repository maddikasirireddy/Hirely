# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0

import os
from typing import List, Literal
from pydantic import BaseModel, Field
from google.adk.agents import Agent
from google.adk.models import Gemini
from google.genai import types
from dotenv import load_dotenv

# Load local environment variables
load_dotenv()

# =====================================================================
# 1. Data Architecture & Schemas (Pydantic Models)
# =====================================================================

class JobProfile(BaseModel):
    title: str = Field(description="The formal title of the role")
    core_technical_challenges: List[str] = Field(
        description="Core complex engineering problems and challenges this role is hired to solve"
    )
    required_trajectory: str = Field(
        description="The ideal career path/growth history expected (e.g. startup scaling, enterprise, specific domain progression)"
    )
    implicitly_expected_traits: List[str] = Field(
        description="Soft skills, working styles, or mindsets that are implicitly expected but not explicitly stated in the job description"
    )
    must_haves: List[str] = Field(
        description="Critical skills, tools, or experiences that are absolute dealbreakers"
    )
    nice_to_haves: List[str] = Field(
        description="Preferred skills or experiences that add value but are not strictly mandatory"
    )


class SkillDepth(BaseModel):
    skill: str = Field(description="Name of the skill or technology")
    level: Literal["Expert", "Intermediate", "Novice"] = Field(
        description="Assessed proficiency level based on project context, duration, and usage complexity"
    )
    context_justification: str = Field(
        description="Detailed evidence/reasoning from the candidate's history justifying this proficiency level"
    )


class CandidateProfile(BaseModel):
    name: str = Field(description="Candidate's full name")
    semantic_trajectory: str = Field(
        description="A summary of the candidate's career velocity, growth, and trajectory patterns (e.g. rapid promotion, specialization, lateral moves)"
    )
    skills_depth: List[SkillDepth] = Field(
        description="List of core skills assessed with their contextual depth"
    )
    behavioral_signals: List[str] = Field(
        description="Signals of ownership, leadership, resilience, mentorship, or adaptability extracted from their project history"
    )
    platform_activity: List[str] = Field(
        description="Open-source contributions, technical writing, public projects, hackathons, or community contributions"
    )


class CandidateEvaluation(BaseModel):
    candidate_name: str = Field(description="Name of the candidate being evaluated")
    technical_fit: float = Field(
        description="Score from 0.0 to 10.0 representing alignment of technical skills and depth with role challenges"
    )
    culture_trajectory_fit: float = Field(
        description="Score from 0.0 to 10.0 representing alignment with expected career trajectory and implicit traits"
    )
    delivery_capability: float = Field(
        description="Score from 0.0 to 10.0 representing execution capability, ownership, and track record of shipping"
    )
    justification: str = Field(
        description="A detailed, qualitative paragraph explaining the reasoning behind the scores and why this candidate fits or struggles with the role"
    )


# =====================================================================
# 2. Agent Definitions
# =====================================================================

# Standard model used for all agents
model_name = "gemini-2.5-flash"
llm = Gemini(
    model=model_name,
    retry_options=types.HttpRetryOptions(attempts=3)
)

# A. Job Understanding Agent
job_understanding_agent = Agent(
    name="job_understanding_agent",
    model=llm,
    instruction=(
        "You are an elite technical recruiter and software architect. Analyze the raw job description "
        "provided by the user and extract its 'True Intent'. Go beyond buzzwords to identify the core technical "
        "challenges the candidate will face daily, the expected career trajectory, implicit soft skills or traits "
        "needed, and differentiate the mandatory requirements (must-haves) from the nice-to-haves. "
        "You must output a structured JSON matching the JobProfile schema."
    ),
    output_schema=JobProfile,
    output_key="job_profile"
)

# B. Candidate Understanding Agent
candidate_understanding_agent = Agent(
    name="candidate_understanding_agent",
    model=llm,
    instruction=(
        "You are an expert technical assessor. Analyze the candidate's resume or career history text "
        "provided by the user. Extract their semantic career trajectory, assess the true depth of their skills "
        "(categorizing them into Expert, Intermediate, or Novice with explicit justification based on how and where "
        "they used each skill), identify behavioral signals (ownership, drive, mentorship), and capture "
        "any open-source or platform activity. You must output a structured JSON matching the CandidateProfile schema."
    ),
    output_schema=CandidateProfile,
    output_key="candidate_profile"
)

# C. Recruiter Reasoning Agent
recruiter_reasoning_agent = Agent(
    name="recruiter_reasoning_agent",
    model=llm,
    instruction=(
        "You are a expert recruiter and talent partner. You are given a structured Job Profile and a "
        "structured Candidate Profile. Evaluate the candidate deeply and objectively against the role requirements. "
        "Look for alignment in skills depth (e.g. does their 'Expert' skill match a 'must-have'?), trajectory "
        "(does their career velocity fit?), and delivery capability. Weigh any career gaps, growth signs, and "
        "behavioral signals. Produce: \n"
        "1. Scores from 0.0 to 10.0 for Technical Fit, Culture/Trajectory Fit, and Delivery Capability.\n"
        "2. A qualitative justification paragraph explaining *why* this candidate fits, noting specific strengths "
        "or potential red flags relative to the core challenges of the job. "
        "You must output a structured JSON matching the CandidateEvaluation schema."
    ),
    output_schema=CandidateEvaluation,
    output_key="candidate_evaluation"
)
