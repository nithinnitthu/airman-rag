"""
Answer Generator — Strictly grounded LLM responses using OpenAI
Enforces hallucination prevention: if context is insufficient, refuses to answer.
"""

import os
import logging
from typing import List, Dict

from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REFUSAL_ANSWER = "This information is not available in the provided document(s)."

SYSTEM_PROMPT = """You are an aviation technical assistant for AIRMAN. Your ONLY job is to answer questions using the aviation document excerpts provided below as context.

STRICT RULES:
1. Answer ONLY from the provided context. Do NOT use any external knowledge.
2. If the context does not contain sufficient information to answer the question, respond EXACTLY with:
   "This information is not available in the provided document(s)."
3. Every factual claim in your answer must be traceable to a specific chunk in the context.
4. Do NOT guess, infer beyond the text, or supplement with general aviation knowledge.
5. Be concise and precise. Aviation safety depends on accuracy.
6. If the question is clearly outside aviation (e.g., about geography, art, history), respond with the refusal message above.

Format your answer as:
ANSWER: <your grounded answer here>
BASIS: <brief note on which part of context supports your answer>
"""


class AnswerGenerator:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def _build_context(self, chunks: List[Dict]) -> str:
        """Format retrieved chunks into a numbered context block."""
        context_parts = []
        for i, chunk in enumerate(chunks, start=1):
            source = chunk.get("source", "unknown")
            page = chunk.get("page", "N/A")
            chunk_id = chunk.get("chunk_id", "?")
            text = chunk.get("text", "")
            context_parts.append(
                f"[CHUNK {i} | Source: {source} | Page: {page} | ID: {chunk_id}]\n{text}"
            )
        return "\n\n---\n\n".join(context_parts)

    def _extract_citations(self, chunks: List[Dict]) -> List[Dict]:
        """Build citation list from top retrieved chunks."""
        citations = []
        seen = set()
        for chunk in chunks:
            key = (chunk.get("source"), chunk.get("page"))
            if key not in seen:
                citations.append({
                    "document": chunk.get("source", "unknown"),
                    "page": chunk.get("page", "N/A"),
                    "chunk_id": chunk.get("chunk_id", "N/A"),
                    "snippet": chunk["text"][:120] + "..." if len(chunk["text"]) > 120 else chunk["text"]
                })
                seen.add(key)
        return citations

    def generate(self, question: str, chunks: List[Dict]) -> Dict:
        """
        Generate a grounded answer from retrieved chunks.
        Returns: { answer, citations, refusal }
        """
        context = self._build_context(chunks)
        citations = self._extract_citations(chunks)

        user_message = f"""CONTEXT FROM AVIATION DOCUMENTS:
{context}

QUESTION: {question}

Remember: Answer ONLY from the context above. If you cannot find the answer there, use the exact refusal phrase."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.0,  # Deterministic for factual accuracy
                max_tokens=600
            )

            raw_answer = response.choices[0].message.content.strip()

            # Parse structured response
            if "ANSWER:" in raw_answer:
                answer = raw_answer.split("ANSWER:")[1].split("BASIS:")[0].strip()
            else:
                answer = raw_answer

            # Detect refusal
            is_refusal = REFUSAL_ANSWER.lower() in answer.lower()

            if is_refusal:
                return {
                    "answer": REFUSAL_ANSWER,
                    "citations": [],
                    "refusal": True
                }

            return {
                "answer": answer,
                "citations": citations,
                "refusal": False
            }

        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return {
                "answer": REFUSAL_ANSWER,
                "citations": [],
                "refusal": True
            }
