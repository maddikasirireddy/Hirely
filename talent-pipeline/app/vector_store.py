# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0

import os
import json
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from app.agents import CandidateProfile, SkillDepth, candidate_understanding_agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import asyncio
from dotenv import load_dotenv

load_dotenv()

# Path to cache structured candidate profiles
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
PROFILES_CACHE_FILE = os.path.join(CACHE_DIR, "candidate_profiles.json")

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

# Initialize the lightweight embedding model
# We use a lazy initializer to avoid loading it when just importing the module
_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        # Load the lightweight and fast SentenceTransformer model
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


# =====================================================================
# 1. Raw Mock Candidate Resumes (The Talent Pool)
# =====================================================================

MOCK_RESUMES = [
    {
        "name": "Alex Chen",
        "resume": """
        Alex Chen - Senior AI/ML Platform Engineer
        Email: alex.chen@example.com | GitHub: github.com/alechen-ml

        Professional Summary:
        AI/ML Systems Engineer with 6+ years of experience building high-performance model training and inference pipelines. Active contributor to PyTorch ecosystem. Specialized in distributed training, LLM serving, and optimizing GPU utilization.

        Experience:
        - Lead ML Platform Engineer at NexusAI (2023 - Present):
          * Architected a distributed training platform using Kubernetes and PyTorch Elastic, reducing training costs by 35% for LLMs.
          * Optimized inference latency of open-source LLMs (Llama, Mistral) from 80ms to 24ms per token using TensorRT-LLM and vLLM.
          * Mentored a team of 4 junior ML engineers; introduced rigorous benchmarking and profiling practices.
        - Senior ML Engineer at Orbit Analytics (2020 - 2023):
          * Built real-time recommendation engines processing 15,000 requests per second.
          * Implemented feature store (Feast) and automated drift detection for 12 production models.
        
        Skills:
        PyTorch (Expert), Python (Expert), C++ (Intermediate), vLLM (Expert), Kubernetes (Expert), CUDA (Intermediate), Triton Inference Server (Expert), Go (Intermediate).

        Open Source & Platform Activity:
        - Maintained 'pytorch-easy-dist', a utility library with 1,200+ stars on GitHub for simplified multi-GPU training setup.
        - Merged 8 PRs into PyTorch core regarding CUDA memory management optimizations.
        """
    },
    {
        "name": "Sarah Jenkins",
        "resume": """
        Sarah Jenkins - Staff Backend & Distributed Systems Engineer
        Email: sarah.j@example.com | GitHub: github.com/sjenkins-dev

        Professional Summary:
        Staff Software Engineer with 10+ years of experience specializing in highly concurrent distributed systems, database internals, and high-throughput microservices. Passionate about clean code, robust testing, and system performance.

        Experience:
        - Staff Engineer at CloudScale Solutions (2021 - Present):
          * Led the architectural redesign of the core messaging backbone, transitioning from a legacy monolith to a Go-based microservice mesh handling 2M+ events per minute.
          * Wrote a custom distributed log storage engine in Go that improved write throughput by 4x.
          * Established engineering standards for testing and reliability; reduced system incidents by 60%.
        - Senior Systems Engineer at FinTech Flow (2017 - 2021):
          * Designed and implemented transaction processing pipelines with strict ACID compliance (handling $50M+ daily).
          * Re-architected PostgreSQL database schemas and connection pooling, reducing query latencies by 50%.
        
        Skills:
        Go (Expert), Python (Expert), PostgreSQL (Expert), Redis (Expert), Kafka (Expert), Distributed Systems (Expert), Docker (Expert), gRPC (Expert), Kubernetes (Intermediate).

        Open Source & Platform Activity:
        - Author of 'golog-db', a transactional append-only log library written in Go (500+ stars).
        - Regular speaker at Go conferences (GopherCon) on distributed consensus algorithms.
        """
    },
    {
        "name": "Elena Rostova",
        "resume": """
        Elena Rostova - Principal Frontend Architect
        Email: elena.r@example.com | GitHub: github.com/elenarostov

        Professional Summary:
        Principal Frontend Engineer with 8+ years of experience designing scalable web architectures, advanced interactive UIs, and robust design systems. Advocate for web accessibility (a11y), performance optimization, and micro-frontends.

        Experience:
        - Principal Frontend Architect at Designify (2022 - Present):
          * Spearheaded the migration of a legacy dashboard to a micro-frontend architecture using Webpack Module Federation, enabling 6 independent teams to deploy concurrently.
          * Built the company's open-source component library used across 14 product lines, focusing on accessibility (WAI-ARIA) and smooth animations.
          * Optimized initial page load times by 45% using code-splitting, tree-shaking, and asset caching strategies.
        - Senior Frontend Engineer at WebFlow Studios (2018 - 2022):
          * Created interactive 3D data visualization tools using React, Three.js, and WebGL.
          * Introduced TypeScript to the codebase, reducing production runtime exceptions by 30%.
        
        Skills:
        React (Expert), TypeScript (Expert), CSS/Sass (Expert), Next.js (Expert), Three.js/WebGL (Intermediate), Webpack/Vite (Expert), Accessibility/a11y (Expert), TailwindCSS (Expert).

        Open Source & Platform Activity:
        - Core contributor to 'accessible-react-primitives', an open-source library for accessible UI elements (2,500+ stars).
        - Wrote a series of technical articles on CSS Houdini and performance profiling in Chrome DevTools.
        """
    },
    {
        "name": "David Kim",
        "resume": """
        David Kim - Product-Minded Full-Stack Developer
        Email: david.k@example.com | GitHub: github.com/dkim-maker

        Professional Summary:
        Entrepreneurial full-stack developer with 5 years of experience. Built and scaled multiple web applications from scratch. Strong product intuition, user empathy, and rapid prototyping capabilities. Winner of multiple national hackathons.

        Experience:
        - Founder & Lead Developer at IndieSaaS (2023 - Present):
          * Conceived, built, and launched 'FormFlow', a micro-SaaS for automated customer feedback collection. Scaled to $8k MRR with 2,000+ active users.
          * Built the entire stack using Next.js, Supabase, TailwindCSS, and Stripe.
          * Executed all product design, customer support, and growth engineering.
        - Full Stack Developer at Promptly (2021 - 2023):
          * Developed collaborative real-time document editing features using WebSockets and Yjs (CRDTs).
          * Integrated Gemini and OpenAI APIs to power AI-assisted brainstorming features.
        
        Skills:
        TypeScript (Expert), React (Expert), Node.js (Expert), Next.js (Expert), PostgreSQL (Intermediate), Supabase (Expert), TailwindCSS (Expert), Python (Intermediate), LLM APIs (Intermediate).

        Open Source & Platform Activity:
        - Created 'svelte-tailwind-boilerplate', a template with 800+ stars.
        - 1st Place Winner at Global GenAI Hackathon 2024 for an AI-powered educational game.
        """
    },
    {
        "name": "Marcus Vance",
        "resume": """
        Marcus Vance - Lead DevSecOps & Infrastructure Engineer
        Email: marcus.v@example.com | GitHub: github.com/marcusv-ops

        Professional Summary:
        Veteran systems engineer specializing in Cloud Infrastructure, DevSecOps, and continuous delivery. Ex-military communications officer. Focus on security compliance (SOC2/HIPAA), infrastructure-as-code, and resilient site reliability.

        Experience:
        - Lead Cloud Security Engineer at SecureHealth (2022 - Present):
          * Designed and implemented a HIPAA-compliant AWS infrastructure using Terraform, achieving zero security breaches and 99.99% uptime.
          * Automated vulnerability scanning (Trivy, SonarQube) in GitLab CI/CD pipelines, catching 95% of security flaws before staging.
          * Managed and automated a 150-node Kubernetes cluster across multiple regions.
        - DevOps Engineer at Core Infrastructure (2019 - 2022):
          * Standardized development environments using Docker and Ansible, reducing developer onboarding time from 3 days to 2 hours.
          * Developed custom Prometheus and Grafana dashboards for real-time application monitoring and alerting.
        
        Skills:
        AWS (Expert), Terraform (Expert), Kubernetes (Expert), Docker (Expert), CI/CD (Expert), Linux Systems (Expert), Python (Intermediate), Bash (Expert), IAM & Security (Expert).

        Open Source & Platform Activity:
        - Created 'terraform-aws-secure-vpc', a highly-rated, secure-by-default Terraform module on GitHub.
        - Contributes to open-source security audit scripts.
        """
    },
    {
        "name": "Aria Patel",
        "resume": """
        Aria Patel - Senior Data Platform Engineer
        Email: aria.p@example.com | GitHub: github.com/ariadata

        Professional Summary:
        Data Engineer with 7 years of experience building scalable ETL/ELT pipelines, real-time data streaming architectures, and data warehouses. Passionate about data quality, cataloging, and optimizing analytics query performance.

        Experience:
        - Senior Data Engineer at ByteData Corp (2022 - Present):
          * Built real-time streaming pipelines using Apache Spark and Kafka, processing 50TB of event data daily.
          * Migrated legacy data warehouse to Snowflake, reducing annual licensing costs by $120k while improving dashboard query speeds by 5x.
          * Implemented dbt (data build tool) for data transformation and automated testing, raising data quality metrics from 78% to 99%.
          * Mentored and trained 5 data analysts and junior data engineers on SQL best practices.
        - Data Engineer at RetailInsights (2019 - 2022):
          * Designed batch ETL pipelines using Apache Airflow and PostgreSQL to aggregate daily sales data.
        
        Skills:
        SQL (Expert), Python (Expert), Snowflake (Expert), Apache Spark (Expert), Apache Airflow (Expert), dbt (Expert), Kafka (Intermediate), Data Modeling (Expert).

        Open Source & Platform Activity:
        - Active contributor to the dbt community; wrote custom macro packages shared on dbt Hub.
        - Organizes local 'Women in Data' meetups and writes articles on modern data stack architectures.
        """
    },
    {
        "name": "Thomas Wright",
        "resume": """
        Thomas Wright - Junior Software Engineer (Self-Taught / Open-Source Contributor)
        Email: thomas.w@example.com | GitHub: github.com/twright-code

        Professional Summary:
        Highly driven, self-taught developer with a deep passion for low-level systems and backend engineering. Active open-source contributor with 2+ years of intensive self-directed learning and project building.

        Experience:
        - Open Source Developer & Freelancer (2024 - Present):
          * Built 'Rust-DB-Lite', a lightweight, transactional key-value store in Rust from scratch to learn database internals (supports ACID, write-ahead logging).
          * Contributed 12 bug fixes and performance improvements to the Hyper HTTP library in Rust.
          * Completed advanced coursework in Algorithms, Operating Systems, and Distributed Systems independently.
          * Designed and hosted a web-based real-time multiplayer chess game using WebSockets and Go.
        
        Skills:
        Rust (Intermediate), Go (Intermediate), Python (Intermediate), SQL (Intermediate), Linux/Bash (Intermediate), Git (Expert), WebSockets (Intermediate).

        Open Source & Platform Activity:
        - Highly active GitHub profile with daily contributions.
        - 'Rust-DB-Lite' repository has 300+ stars and several external contributors.
        """
    },
    {
        "name": "Sophia Martinez",
        "resume": """
        Sophia Martinez - Senior Backend Engineer (Returning to Tech / AI Specialization)
        Email: sophia.m@example.com | GitHub: github.com/smartinez-dev

        Professional Summary:
        Experienced backend software engineer with 10 years of professional experience leading enterprise Java systems. Returning to the tech industry after a 3-year sabbatical for family care, during which I completed comprehensive retraining in modern Python, generative AI integration, and cloud-native systems.

        Experience:
        - AI Integration & Advanced Training (Sabbatical/Retraining) (2023 - 2026):
          * Completed deep-dive courses in Generative AI, PyTorch, and LLM orchestration.
          * Built several portfolio applications including an AI-powered document summarizer using LangChain and a custom RAG (Retrieval-Augmented Generation) pipeline over legal documents.
        - Tech Lead / Senior Java Engineer at Enterprise Solutions (2013 - 2023):
          * Led a team of 6 engineers maintaining a high-availability Spring Boot backend for an e-commerce platform (serving 1M+ daily active users).
          * Managed database migrations, API design, and integrations with legacy SOAP services.
          * Awarded 'Employee of the Year' in 2019 for successfully leading a zero-downtime cloud migration.
        
        Skills:
        Java (Expert), Spring Boot (Expert), Python (Intermediate), SQL (Expert), REST APIs (Expert), AWS (Intermediate), RAG/Vector Databases (Intermediate), PyTorch (Novice).

        Open Source & Platform Activity:
        - Published a comprehensive blog series detailing her journey returning to tech, sharing learning resources and RAG tutorials.
        """
    },
    {
        "name": "Liam O'Connor",
        "resume": """
        Liam O'Connor - Technical Product Lead & Ex-Developer
        Email: liam.oc@example.com | GitHub: github.com/loconnor-prod

        Professional Summary:
        Hybrid technical leader with 4 years of software development experience followed by 4 years of Product Management and Agile team leadership. Expert at bridging the gap between business strategy and deep technical execution.

        Experience:
        - Technical Product Manager / Team Lead at InnovateCorp (2022 - Present):
          * Product owner for the Developer Platform team. Defined product roadmap, reduced CI/CD build times by 40% as a core product metric.
          * Acted as Agile Scrum Master, improving team velocity by 25% through improved planning and bottleneck identification.
          * Designed API contracts and collaborated closely with software architects to ensure scalable system design.
        - Full-Stack Software Engineer at InnovateCorp (2018 - 2022):
          * Developed frontend and backend features for an enterprise analytics dashboard using React, Node.js, and MongoDB.
          * Led the integration of third-party Salesforce and HubSpot APIs.
        
        Skills:
        Agile/Scrum (Expert), Product Strategy (Expert), API Design (Expert), System Architecture (Intermediate), JavaScript/Node.js (Expert), SQL (Intermediate), Git (Expert).

        Open Source & Platform Activity:
        - Wrote open-source product management templates and guides for GitHub.
        - Mentors ex-developers transitioning into product management roles.
        """
    }
]


# =====================================================================
# 2. Candidate Parsing & Indexing Logic
# =====================================================================

def serialize_profile(profile: CandidateProfile) -> Dict[str, Any]:
    """Helper to serialize Pydantic CandidateProfile to dict."""
    return {
        "name": profile.name,
        "semantic_trajectory": profile.semantic_trajectory,
        "skills_depth": [
            {
                "skill": s.skill,
                "level": s.level,
                "context_justification": s.context_justification
            } for s in profile.skills_depth
        ],
        "behavioral_signals": profile.behavioral_signals,
        "platform_activity": profile.platform_activity
    }

def deserialize_profile(data: Dict[str, Any]) -> CandidateProfile:
    """Helper to deserialize dict to Pydantic CandidateProfile."""
    skills = [
        SkillDepth(
            skill=s["skill"],
            level=s["level"],
            context_justification=s["context_justification"]
        ) for s in data["skills_depth"]
    ]
    return CandidateProfile(
        name=data["name"],
        semantic_trajectory=data["semantic_trajectory"],
        skills_depth=skills,
        behavioral_signals=data["behavioral_signals"],
        platform_activity=data["platform_activity"]
    )

async def build_candidate_profiles(progress_callback=None) -> List[CandidateProfile]:
    """Runs CandidateUnderstandingAgent on mock resumes if cache is missing, saving to cache."""
    if os.path.exists(PROFILES_CACHE_FILE):
        if progress_callback:
            progress_callback("📦 Loading candidate profiles from local cache...")
        with open(PROFILES_CACHE_FILE, "r") as f:
            cached_data = json.load(f)
            return [deserialize_profile(p) for p in cached_data]

    if progress_callback:
        progress_callback("🤖 Cache missing. Running Candidate Understanding Agent on mock resumes...")

    profiles = []
    session_service = InMemorySessionService()

    # Process candidates sequentially (or in parallel if desired, keeping it safe here)
    for idx, candidate in enumerate(MOCK_RESUMES):
        name = candidate["name"]
        resume_text = candidate["resume"]

        if progress_callback:
            progress_callback(f"🧠 Parsing resume of {name} ({idx+1}/{len(MOCK_RESUMES)})...")

        # Create a session for this parsing task
        session_id = f"parse_{name.lower().replace(' ', '_')}"
        await session_service.create_session(app_name="app", user_id="recruiter", session_id=session_id)
        
        # Instantiate runner for candidate agent
        runner = Runner(
            agent=candidate_understanding_agent,
            app_name="app",
            session_service=session_service
        )

        # Run candidate agent
        async for _ in runner.run_async(
            user_id="recruiter",
            session_id=session_id,
            new_message=types.Content(role="user", parts=[types.Part.from_text(text=resume_text)])
        ):
            pass

        # Retrieve structured profile from state
        session = await session_service.get_session(app_name="app", user_id="recruiter", session_id=session_id)
        profile_raw = session.state.get("candidate_profile")
        
        if profile_raw:
            if isinstance(profile_raw, dict):
                profile = CandidateProfile(**profile_raw)
            else:
                profile = profile_raw
            profiles.append(profile)
        else:
            # Fallback if agent failed (should not happen, but for safety)
            fallback = CandidateProfile(
                name=name,
                semantic_trajectory="Experienced professional.",
                skills_depth=[SkillDepth(skill="Software Engineering", level="Intermediate", context_justification="Based on resume.")],
                behavioral_signals=["Professional"],
                platform_activity=["None"]
            )
            profiles.append(fallback)

    # Save to cache
    serialized = [serialize_profile(p) for p in profiles]
    with open(PROFILES_CACHE_FILE, "w") as f:
        json.dump(serialized, f, indent=2)

    if progress_callback:
        progress_callback("✅ Candidate profiles generated and cached successfully!")

    return profiles


def get_candidate_text_for_embedding(profile: CandidateProfile) -> str:
    """Combines candidate attributes into a single semantically rich text block for embedding."""
    skills_str = ", ".join([f"{s.skill} ({s.level}: {s.context_justification})" for s in profile.skills_depth])
    traits_str = ", ".join(profile.behavioral_signals)
    activity_str = ", ".join(profile.platform_activity)
    
    text = (
        f"Role Trajectory: {profile.semantic_trajectory}\n"
        f"Key Skills: {skills_str}\n"
        f"Behavioral Traits: {traits_str}\n"
        f"Open Source Projects: {activity_str}"
    )
    return text


class CandidateVectorStore:
    """Manages the local FAISS index and candidate retrieval."""
    
    def __init__(self):
        self.profiles: List[CandidateProfile] = []
        self.index = None
        self.model = None

    async def initialize(self, progress_callback=None):
        """Initializes profiles, loads embedding model, and builds FAISS index."""
        # 1. Build or load structured profiles
        self.profiles = await build_candidate_profiles(progress_callback)
        
        # 2. Initialize embedding model
        if progress_callback:
            progress_callback("📡 Loading sentence-transformer embedding model (all-MiniLM-L6-v2)...")
        self.model = get_embedding_model()
        
        # 3. Build FAISS index
        if progress_callback:
            progress_callback("⚡ Building local FAISS vector index...")
        
        texts = [get_candidate_text_for_embedding(p) for p in self.profiles]
        embeddings = self.model.encode(texts, show_progress_bar=False)
        
        dimension = embeddings.shape[1]
        # Using IndexFlatIP (Inner Product) with normalized vectors for Cosine Similarity
        self.index = faiss.IndexFlatIP(dimension)
        
        # Normalize embeddings
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        if progress_callback:
            progress_callback(f"🎉 FAISS index loaded with {self.index.ntotal} candidate profiles!")

    def retrieve(self, job_profile: Any, top_k: int = 5) -> List[Tuple[CandidateProfile, float]]:
        """Retrieves top_k candidates matching a structured job profile, returning (profile, similarity_score)."""
        if self.index is None or self.model is None:
            raise ValueError("Vector store is not initialized. Call initialize() first.")

        # Construct job description query text
        query_text = (
            f"Title: {job_profile.title}\n"
            f"Challenges: {', '.join(job_profile.core_technical_challenges)}\n"
            f"Trajectory: {job_profile.required_trajectory}\n"
            f"Must haves: {', '.join(job_profile.must_haves)}\n"
            f"Nice to haves: {', '.join(job_profile.nice_to_haves)}"
        )
        
        # Embed query and normalize
        query_vector = self.model.encode([query_text])
        faiss.normalize_L2(query_vector)
        
        # Search index
        scores, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            # FAISS IndexFlatIP returns cosine similarity in [-1, 1] for L2 normalized vectors
            # Convert to [0, 1] range for friendly display
            normalized_score = float((score + 1.0) / 2.0)
            results.append((self.profiles[idx], normalized_score))
            
        return results

    async def add_candidate(self, name: str, resume_text: str) -> CandidateProfile:
        """Parses a new candidate on the fly, checking the cache first before using Gemini."""
        # 1. Check if candidate is already parsed (in-memory or in cache JSON)
        existing_profile = None
        
        # Check in-memory
        for p in self.profiles:
            if p.name.strip().lower() == name.strip().lower():
                existing_profile = p
                break
                
        # Check cache file
        if not existing_profile and os.path.exists(PROFILES_CACHE_FILE):
            with open(PROFILES_CACHE_FILE, "r") as f:
                try:
                    serialized = json.load(f)
                    for item in serialized:
                        if item.get("name", "").strip().lower() == name.strip().lower():
                            existing_profile = deserialize_profile(item)
                            # Add to in-memory profiles
                            self.profiles.append(existing_profile)
                            # Add to FAISS index
                            text = get_candidate_text_for_embedding(existing_profile)
                            vector = self.model.encode([text])
                            faiss.normalize_L2(vector)
                            self.index.add(vector)
                            break
                except Exception:
                    pass

        if existing_profile:
            return existing_profile

        # 2. If No -> Parse using Gemini agent
        session_service = InMemorySessionService()
        session_id = f"parse_{name.lower().replace(' ', '_')}_{int(asyncio.get_event_loop().time())}"
        await session_service.create_session(app_name="app", user_id="recruiter", session_id=session_id)
        
        runner = Runner(
            agent=candidate_understanding_agent,
            app_name="app",
            session_service=session_service
        )

        async for _ in runner.run_async(
            user_id="recruiter",
            session_id=session_id,
            new_message=types.Content(role="user", parts=[types.Part.from_text(text=resume_text)])
        ):
            pass

        session = await session_service.get_session(app_name="app", user_id="recruiter", session_id=session_id)
        profile_raw = session.state.get("candidate_profile")
        
        if not profile_raw:
            raise ValueError(f"Failed to parse candidate resume for {name}")

        if isinstance(profile_raw, dict):
            profile = CandidateProfile(**profile_raw)
        else:
            profile = profile_raw

        # 3. Add to active profiles
        self.profiles.append(profile)
        
        # 4. Save to cache file
        serialized = []
        if os.path.exists(PROFILES_CACHE_FILE):
            with open(PROFILES_CACHE_FILE, "r") as f:
                try:
                    serialized = json.load(f)
                except Exception:
                    serialized = []
        serialized.append(serialize_profile(profile))
        with open(PROFILES_CACHE_FILE, "w") as f:
            json.dump(serialized, f, indent=2)
            
        # 5. Add to FAISS index
        text = get_candidate_text_for_embedding(profile)
        vector = self.model.encode([text])
        faiss.normalize_L2(vector)
        self.index.add(vector)
        
        return profile

    def remove_candidate(self, name: str) -> bool:
        """Removes a candidate by name from profiles, cache file, and rebuilds the FAISS index.
        
        Returns True if the candidate was found and removed, False otherwise.
        FAISS IndexFlatIP does not support in-place deletion, so the index is fully
        rebuilt from the remaining profiles after removal.
        """
        # Find the candidate
        original_count = len(self.profiles)
        self.profiles = [p for p in self.profiles if p.name != name]
        
        if len(self.profiles) == original_count:
            return False  # Not found

        # Persist updated profiles to cache
        serialized = [serialize_profile(p) for p in self.profiles]
        with open(PROFILES_CACHE_FILE, "w") as f:
            json.dump(serialized, f, indent=2)

        # Rebuild FAISS index from remaining profiles
        if self.profiles:
            texts = [get_candidate_text_for_embedding(p) for p in self.profiles]
            embeddings = self.model.encode(texts, show_progress_bar=False)
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings)
        else:
            # No profiles left — create empty index with the same dimension
            dummy = self.model.encode(["placeholder"])
            self.index = faiss.IndexFlatIP(dummy.shape[1])

        return True
