from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any, Dict, List, Optional

import logging

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from openai import OpenAI
from pydantic import BaseModel, Field

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_PROJECT_DIR, ".env"))

logger = logging.getLogger(__name__)

app = FastAPI()

_INDEX_PATH = os.path.join(_PROJECT_DIR, "index.html")

_client: Optional[OpenAI] = None


def _get_client() -> Optional[OpenAI]:
    """Create the LLM client lazily so the app still starts when no API key is set."""
    global _client
    if _client is None:
        api_key = (os.getenv("OPENAI_API_KEY") or os.getenv("FIREWORKS_API_KEY") or "").strip()
        if not api_key:
            return None
        _client = OpenAI(
            api_key=api_key,
            base_url=(os.getenv("OPENAI_BASE_URL") or "https://api.fireworks.ai/inference/v1").strip(),
        )
    return _client


class GenerateTaskRequest(BaseModel):
    subjects: List[str] = Field(min_length=2, max_length=10)
    chosen_subject: str = Field(min_length=1)
    hardcore: bool = False


@app.get("/")
async def home() -> FileResponse:
    return FileResponse(_INDEX_PATH)


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/generate")
async def api_generate(req: GenerateTaskRequest) -> JSONResponse:
    subjects = [s.strip() for s in req.subjects if s and s.strip()][:10]
    if len(subjects) < 2:
        return JSONResponse({"error": "Please provide at least 2 subjects."}, status_code=400)

    chosen = req.chosen_subject.strip()
    if chosen not in subjects:
        return JSONResponse({"error": "Chosen subject must be one of the provided subjects."}, status_code=400)

    task = await asyncio.to_thread(_generate_study_task, subjects, chosen, req.hardcore)
    if task is None:
        return JSONResponse({"error": "no_connection", "message": "No connection"}, status_code=503)
    return JSONResponse(task)


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _is_generic_topic(topic: str) -> bool:
    t = topic.lower()
    return t in {"core concepts", "general review", "basics", ""}


def _is_generic_task(task: str) -> bool:
    low = task.lower()
    return not task.strip() or any(
        p in low
        for p in [
            "do exercises",
            "solve problems",
            "practice questions",
            "review notes",
        ]
    )


def _pack_api_task(parsed: Dict[str, Any], chosen_subject: str, hardcore: bool) -> Optional[Dict[str, str]]:
    subject_api = str(parsed.get("subject") or "").strip()
    if subject_api.casefold() != chosen_subject.casefold():
        return None

    topic = str(parsed.get("topic") or "").strip()
    time_s = str(parsed.get("time") or "").strip()
    task = str(parsed.get("task") or "").strip()
    why = str(parsed.get("why_important") or "").strip()
    hardcore_rule = str(parsed.get("hardcore_rule") or "").strip()

    if not topic or not time_s or not task or not why:
        return None
    if _is_generic_topic(topic) or _is_generic_task(task):
        return None

    if hardcore and not hardcore_rule:
        return None

    return {
        "subject": chosen_subject,
        "topic": topic,
        "time": time_s,
        "task": task,
        "why_important": why,
        "hardcore_rule": hardcore_rule if hardcore else "",
    }


def _generate_study_task(subjects: List[str], chosen_subject: str, hardcore: bool) -> Optional[Dict[str, str]]:
    time_options = ["20 minutes", "30 minutes", "45 minutes", "60 minutes"]
    intensity = "hardcore" if hardcore else "normal"

    prompt = f"""
You are a study coach. Create ONE study task that is concrete and specific (not generic).

User subjects: {", ".join(subjects)}
Chosen subject (MUST use): {chosen_subject}
Mode: {intensity}

You MUST:
- pick a concrete subtopic inside "{chosen_subject}" (not the subject name itself)
- choose a task type that matches the subtopic (build, analyze, write, debug, design, experiment, teach, flashcards, concept map, mini-project, etc.)
- give a task with clear steps and a deliverable/output (something the user produces)
- if the chosen subject is a TOOL/LIBRARY (e.g. Pandas), the task should involve using it on a realistic artifact (data/code), not “do exercises”

Return ONLY valid JSON with exactly these keys:
{{
  "subject": "{chosen_subject}",
  "topic": "a specific subtopic inside the subject",
  "time": "one of: {", ".join(time_options)}",
  "task": "a concrete, actionable task tied to the topic with steps + a deliverable",
  "why_important": "1-2 sentences, practical motivation",
  "hardcore_rule": "if Mode is hardcore, a strict rule; otherwise an empty string"
}}

Rules:
- subject MUST equal "{chosen_subject}"
- topic must be specific (e.g., "Bayes theorem", "Python list comprehensions", "Cell respiration", "Newton's 2nd law", "Quadratic formula", "French Revolution: causes")
- DO NOT give a generic task like "do exercises" / "solve problems" / "review notes"
- task must explicitly reference the topic and include a deliverable
- keep it concise and specific (no fluff)
- time MUST be one of the allowed options

Example of the quality bar (DO NOT copy verbatim; use the user's chosen subject):
{{
  "subject": "Pandas",
  "topic": "Data cleaning (missing values + duplicates)",
  "time": "45 minutes",
  "task": "Download a messy CSV (any public dataset). In a new notebook, load it with pandas, profile missingness, then: (1) drop rows where the target is null, (2) impute a numeric column with median, (3) standardize one text column (strip/lower), (4) remove duplicates, (5) export cleaned.csv. Deliverable: cleaned.csv + 6-line summary of what changed.",
  "why_important": "Real-world data is messy. Cleaning it correctly prevents downstream analysis bugs and improves model/insight reliability.",
  "hardcore_rule": ""
}}
""".strip()

    client = _get_client()
    if client is None:
        logger.warning("No API key set. Add OPENAI_API_KEY to .env (see .env.example).")
        return None

    # Default: serverless model that is currently available on Fireworks (v3p1 70B often 404s).
    model = os.getenv(
        "OPENAI_MODEL",
        "accounts/fireworks/models/llama-v3p3-70b-instruct",
    ).strip()

    def call_llm(extra_instruction: str = "") -> Optional[Dict[str, Any]]:
        messages = [
            {
                "role": "system",
                "content": "You output strictly valid JSON only. No markdown. No commentary.",
            },
            {"role": "user", "content": prompt + (("\n\n" + extra_instruction) if extra_instruction else "")},
        ]
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.6,
                max_tokens=768,
            )
        except Exception as exc:
            logger.warning("LLM request failed: %s", exc)
            return None
        content = (response.choices[0].message.content or "").strip()
        return _extract_first_json_object(content)

    parsed = call_llm()
    if not parsed:
        parsed = call_llm("Retry: output ONLY the JSON object. Do not wrap it in text.")
    if not parsed:
        return None

    packed = _pack_api_task(parsed, chosen_subject, hardcore)
    if packed:
        return packed

    parsed2 = call_llm(
        "Your last attempt was too generic or invalid. "
        f'Ensure subject == "{chosen_subject}". Pick a specific subtopic and a concrete task with a deliverable. '
        f'If Mode is hardcore, hardcore_rule must be a non-empty strict rule; otherwise hardcore_rule must be "".'
    )
    if not parsed2:
        return None
    return _pack_api_task(parsed2, chosen_subject, hardcore)
