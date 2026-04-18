import asyncio
import json
import logging
import re
import time
from typing import Any, List, Optional, Tuple

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config.settings import get_settings
from schemas.requests import Citation, CritiqueIssue, PageContent, PageSuggestion
from services.chunking_service import Chunk

logger = logging.getLogger(__name__)


class LLMService:
    """Service for LLM-based summarization with citation support"""

    def __init__(self):
        self.settings = get_settings()
        self.models = self._init_models()
        self._last_task_kind = "answer"
        self._last_provider_name = self.settings.resolved_cloud_provider()
        self._last_provider_mode = self.settings.llm_provider
        self._last_model_name = self.settings.resolved_summary_model()
        self._last_model_slot = "summary"
        self._last_model_fallback_used = False
        self._last_routing_note = "Used the summary route for a lightweight answer."
        self._last_generation_metrics: dict[str, Any] = {}
        self._last_prefilter_metrics: dict[str, Any] = {}
        logger.info(
            "LLM client initialized provider=%s model=%s reasoning_model=%s base_url=%s reasoning_effort=%s timeout_seconds=%s",
            self.settings.llm_provider,
            self.settings.resolved_summary_model(),
            self.settings.resolved_reasoning_model(),
            self.settings.resolved_llm_base_url(),
            self.settings.openai_reasoning_effort or None,
            self._timeout_seconds_for_task("answer"),
        )

    def _normalize_task_kind(self, task_kind: str | None) -> str:
        normalized = (task_kind or "answer").strip().lower()
        return normalized if normalized in {"answer", "critique", "diagram"} else "answer"

    def _model_slot_for_task(self, task_kind: str | None) -> str:
        normalized = self._normalize_task_kind(task_kind)
        if normalized in {"critique", "diagram"}:
            return "reasoning"
        return "summary"

    def _supports_custom_temperature(self, model_name: str) -> bool:
        normalized = (model_name or "").strip().lower()

        # Reasoning-oriented OpenAI-compatible models often only accept the
        # provider default temperature. Leave it unset for those families.
        if (
            normalized.startswith("o1") or
            normalized.startswith("o3") or
            normalized.startswith("o4") or
            normalized.startswith("gpt-5") or
            "codex" in normalized
        ):
            return False

        return True

    def _build_openai_model(self, model_name: str):
        api_key = self.settings.resolved_openai_api_key()
        if not api_key:
            raise ValueError("OpenAI API key or PAT token not found in settings")

        init_kwargs = dict(
            api_key=api_key,
            model=model_name,
            base_url=self.settings.resolved_openai_base_url(),
            max_tokens=self.settings.max_output_tokens,
            timeout=self.settings.request_timeout_seconds,
            max_retries=1,
        )

        if self._supports_custom_temperature(model_name):
            init_kwargs["temperature"] = 0.3
        else:
            init_kwargs["temperature"] = 1

        if self.settings.openai_reasoning_effort:
            init_kwargs["reasoning_effort"] = self.settings.openai_reasoning_effort

        return ChatOpenAI(**init_kwargs)

    def _build_ollama_model(self, model_name: str):
        if not model_name.strip():
            raise ValueError("Ollama model name is required")

        return ChatOpenAI(
            api_key="ollama",
            model=model_name,
            base_url=self.settings.resolved_ollama_openai_base_url(),
            max_tokens=self.settings.max_output_tokens,
            timeout=self._timeout_seconds_for_task("answer"),
            max_retries=0,
            temperature=0.3,
        )

    def _build_anthropic_model(self, model_name: str):
        if not self.settings.anthropic_api_key:
            raise ValueError("Anthropic API key not found in settings")

        return ChatAnthropic(
            api_key=self.settings.anthropic_api_key,
            model=model_name,
            temperature=0.3,
            max_tokens=self.settings.max_output_tokens,
            timeout=self.settings.request_timeout_seconds,
            max_retries=1,
        )

    def _build_model_for_provider(self, provider: str, model_name: str):
        if provider == "openai":
            return self._build_openai_model(model_name)
        if provider == "anthropic":
            return self._build_anthropic_model(model_name)
        if provider == "ollama":
            return self._build_ollama_model(model_name)
        raise ValueError(f"Unknown LLM provider: {provider}")

    def _init_models(self) -> dict[str, dict[str, Any]]:
        """Initialize provider clients, optionally splitting summary vs reasoning models."""
        try:
            active_provider = self.settings.resolved_cloud_provider()

            if active_provider in {"openai", "anthropic", "ollama"} and not self.settings.hybrid_mode_enabled():
                summary_model_name = self.settings.resolved_summary_model()
                reasoning_model_name = self.settings.resolved_reasoning_model()
                summary_client = self._build_model_for_provider(active_provider, summary_model_name)
                reasoning_client = (
                    summary_client
                    if reasoning_model_name == summary_model_name
                    else self._build_model_for_provider(active_provider, reasoning_model_name)
                )
                return {
                    "summary": {
                        "client": summary_client,
                        "model_name": summary_model_name,
                        "provider": active_provider,
                        "configured": True,
                    },
                    "reasoning": {
                        "client": reasoning_client,
                        "model_name": reasoning_model_name,
                        "provider": active_provider,
                        "configured": self._is_reasoning_model_configured(active_provider),
                    },
                }

            if self.settings.hybrid_mode_enabled():
                cloud_provider = self.settings.resolved_cloud_provider()
                summary_model_name = self.settings.resolved_summary_model()
                reasoning_model_name = self.settings.resolved_reasoning_model()
                summary_client = self._build_model_for_provider(cloud_provider, summary_model_name)
                reasoning_client = (
                    summary_client
                    if reasoning_model_name == summary_model_name
                    else self._build_model_for_provider(cloud_provider, reasoning_model_name)
                )
                prefilter_model_name = self.settings.resolved_prefilter_model()
                prefilter_client = self._build_ollama_model(prefilter_model_name)
                return {
                    "summary": {
                        "client": summary_client,
                        "model_name": summary_model_name,
                        "provider": cloud_provider,
                        "configured": True,
                    },
                    "reasoning": {
                        "client": reasoning_client,
                        "model_name": reasoning_model_name,
                        "provider": cloud_provider,
                        "configured": self._is_reasoning_model_configured(cloud_provider),
                    },
                    "prefilter": {
                        "client": prefilter_client,
                        "model_name": prefilter_model_name,
                        "provider": "ollama",
                        "configured": bool(prefilter_model_name.strip()) and self.settings.hybrid_prefilter_enabled,
                    },
                }

            raise ValueError(f"Unknown LLM provider: {self.settings.llm_provider}")
        except Exception:
            logger.exception(
                "Failed to initialize LLM client provider=%s model=%s reasoning_model=%s base_url=%s has_pat_token=%s has_openai_api_key=%s",
                self.settings.llm_provider,
                self.settings.resolved_summary_model(),
                self.settings.resolved_reasoning_model(),
                self.settings.resolved_llm_base_url(),
                bool(self.settings.pat_token),
                bool(self.settings.openai_api_key),
            )
            raise

    def _is_reasoning_model_configured(self, provider: str) -> bool:
        if provider == "openai":
            return bool(self.settings.openai_reasoning_model.strip())
        if provider == "anthropic":
            return bool(self.settings.anthropic_reasoning_model.strip())
        if provider == "ollama":
            return bool(self.settings.ollama_reasoning_model.strip())
        return False

    def _get_model_for_task(self, task_kind: str | None) -> tuple[Any, str, str, bool, str, str]:
        slot = self._model_slot_for_task(task_kind)
        descriptor = self.models.get(slot)

        if descriptor is None:
            summary_descriptor = self.models["summary"]
            return (
                summary_descriptor["client"],
                summary_descriptor["model_name"],
                summary_descriptor.get("provider", self.settings.resolved_cloud_provider()),
                "summary",
                True,
                f"Requested {slot} routing, but only the summary model was available.",
            )

        fallback_used = slot == "reasoning" and not bool(descriptor.get("configured"))
        if slot == "reasoning":
            routing_note = (
                "No dedicated reasoning model was configured, so the summary model handled this task."
                if fallback_used
                else "Used the reasoning route for critique/diagram style work."
            )
        else:
            routing_note = "Used the summary route for a lightweight answer."

        return (
            descriptor["client"],
            descriptor["model_name"],
            descriptor.get("provider", self.settings.resolved_cloud_provider()),
            slot,
            fallback_used,
            routing_note,
        )

    def _timeout_seconds_for_task(self, task_kind: str | None) -> int:
        if self.settings.llm_provider == "ollama":
            return max(1, min(self.settings.request_timeout_seconds, self.settings.ollama_request_timeout_seconds))
        return self.settings.request_timeout_seconds

    def _build_llm_usage_metadata(self) -> dict[str, Any]:
        metadata = {
            "provider_used": self._last_provider_name,
            "provider_mode": self._last_provider_mode,
            "model_used": self._last_model_name,
            "model_route": self._last_model_slot,
            "model_fallback_used": self._last_model_fallback_used,
            "model_routing_note": self._last_routing_note,
        }
        metadata.update(self._last_generation_metrics)
        return metadata

    def _find_numeric_metadata_value(self, value: Any, target_keys: set[str]) -> float | None:
        if isinstance(value, dict):
            for key, nested_value in value.items():
                if key in target_keys and isinstance(nested_value, (int, float)):
                    return float(nested_value)
                found = self._find_numeric_metadata_value(nested_value, target_keys)
                if found is not None:
                    return found
        elif isinstance(value, list):
            for item in value:
                found = self._find_numeric_metadata_value(item, target_keys)
                if found is not None:
                    return found
        return None

    def _extract_generation_metrics(self, response: Any, wall_clock_ms: int) -> dict[str, Any]:
        response_metadata = getattr(response, "response_metadata", {}) or {}
        additional_kwargs = getattr(response, "additional_kwargs", {}) or {}
        combined_metadata = {
            "response_metadata": response_metadata,
            "additional_kwargs": additional_kwargs,
        }

        ollama_total_ns = self._find_numeric_metadata_value(
            combined_metadata,
            {"total_duration"},
        )
        ollama_load_ns = self._find_numeric_metadata_value(
            combined_metadata,
            {"load_duration"},
        )
        ollama_prompt_eval_ns = self._find_numeric_metadata_value(
            combined_metadata,
            {"prompt_eval_duration"},
        )
        ollama_eval_ns = self._find_numeric_metadata_value(
            combined_metadata,
            {"eval_duration"},
        )
        explicit_ttft_ms = self._find_numeric_metadata_value(
            combined_metadata,
            {"ttft_ms", "time_to_first_token_ms", "first_token_ms"},
        )

        metrics: dict[str, Any] = {
            "generation_latency_ms": wall_clock_ms,
            "timing_source": "wall-clock",
        }

        if explicit_ttft_ms is not None:
            metrics["ttft_ms"] = round(explicit_ttft_ms)
            metrics["timing_source"] = "provider-ttft"
            return metrics

        if ollama_total_ns is not None:
            metrics["provider_total_latency_ms"] = round(ollama_total_ns / 1_000_000)
            metrics["timing_source"] = "ollama-provider"

        if ollama_eval_ns is not None:
            metrics["provider_eval_latency_ms"] = round(ollama_eval_ns / 1_000_000)

        if ollama_load_ns is not None:
            metrics["provider_load_latency_ms"] = round(ollama_load_ns / 1_000_000)

        if ollama_prompt_eval_ns is not None:
            metrics["provider_prompt_eval_latency_ms"] = round(ollama_prompt_eval_ns / 1_000_000)

        if ollama_load_ns is not None or ollama_prompt_eval_ns is not None:
            ttft_ns = (ollama_load_ns or 0) + (ollama_prompt_eval_ns or 0)
            if ttft_ns > 0:
                metrics["ttft_ms"] = round(ttft_ns / 1_000_000)

        return metrics

    def _build_system_prompt(self) -> str:
        """Build system prompt for summarization with context awareness"""
        return """You are a helpful assistant that synthesizes information from multiple web pages.

Your task is to:
1. Read and understand content from multiple sources
2. Provide clear, accurate summaries
3. Always cite your sources by referencing page titles
4. Highlight any contradictions between sources
5. Focus on factual information from the documents
6. **Detect when information is missing** - if the question asks about something not in the pinned pages, mention it

Response format:
- Start with a clear overview
- Use bullet points for key information
- Add "Source: [Page Title]" after each claim
- Note any contradictions or conflicting information
- Be concise but comprehensive

**Missing Context Detection:**
If the user asks about topics/versions/features NOT covered in the available documents:
- State what information is missing
- Suggest what types of pages might contain that information
- Use this format at the END of your response:

---
💡 **Missing Information:**
The question mentions [X], but I don't see information about [X] in the pinned pages. You might want to search for and pin pages about:
- [Topic A] - [brief reason]
- [Topic B] - [brief reason]

Keywords to search: [keyword1], [keyword2], [keyword3]
---

Only include this section if there are clear gaps in available information."""

    def _build_diagram_system_prompt(self, diagram_type: Optional[str] = None) -> str:
        """Build system prompt for Mermaid diagram generation"""
        diagram_hint = (
            f"Prefer this Mermaid diagram type when it fits the request: {diagram_type}.\n"
            if diagram_type
            else ""
        )

        return f"""You are a helpful assistant that synthesizes information from multiple web pages and turns them into Mermaid diagrams.

Your task is to:
1. Read the provided excerpts carefully and identify the core flow, architecture, or interaction pattern
2. Return a short explanation first (1-3 sentences)
3. Then output exactly one Mermaid diagram inside a fenced ```mermaid code block
4. Choose the best format automatically:
   - sequenceDiagram for request/response or actor interactions
   - flowchart TD for process flows
   - graph TD for architecture or dependency maps
   - classDiagram only when the content clearly describes data models
5. Keep labels short and readable
6. Do not invent systems, steps, or relationships that are not supported by the provided context
7. If the information is incomplete, say what is missing before the diagram and keep the diagram conservative
8. After the diagram, add 2-4 bullet points with the key takeaways and reference source page titles inline

Important Mermaid rules:
- Output valid Mermaid syntax only inside the fenced block
- Avoid parentheses in node ids
- Prefer simple node labels over long sentences
- Do not output more than one Mermaid block
{diagram_hint}"""

    def _build_critic_system_prompt(self) -> str:
        """Build system prompt for requirement critique"""
        return """You are a senior product and systems reviewer.

Review the provided document excerpts for missing requirements, edge cases, logical gaps, and risky assumptions.
Prioritize issues that would matter to engineers, QA, and planners.

Focus on:
- reliability and failure handling
- state transitions and edge cases
- security and permissions
- validation and data integrity
- concurrency, race conditions, and retries
- observability and operational gaps
- unclear ownership, dependencies, or acceptance criteria

Return JSON only with this exact shape:
{
  "summary": "2-4 sentence overview",
  "issues": [
    {
      "title": "short issue title",
      "severity": "high" | "medium" | "low",
      "category": "reliability | security | validation | performance | UX | data | process | dependency | observability | other",
      "evidence": "short exact quote or close paraphrase from the source",
      "risk": "why this matters",
      "suggestion": "specific mitigation or follow-up",
      "source_title": "best matching page title"
    }
  ]
}

Rules:
- Return at most 5 issues
- Use an empty array if there are no material issues
- Be conservative and do not invent unsupported facts
- Prefer precise, implementation-relevant findings over generic advice"""

    def _is_diagram_request(self, user_question: Optional[str]) -> bool:
        """Detect whether the user is asking for a visualized response"""
        if not user_question:
            return False

        question = user_question.lower()
        keywords = [
            "diagram",
            "flowchart",
            "flow chart",
            "sequence",
            "architecture",
            "visualize",
            "visualise",
            "mermaid",
            "show me the flow",
            "show the flow",
            "draw",
            "graph"
        ]

        return any(keyword in question for keyword in keywords)

    def _build_context(self, pages: List[PageContent]) -> str:
        """Build context from multiple pages with clear source labels"""
        context_parts = []

        for idx, page in enumerate(pages, 1):
            source_type = page.source_type.upper()
            context_parts.append(
                f"[SOURCE {idx}: {source_type}]\n"
                f"Title: {page.title}\n"
                f"URL: {page.url}\n\n"
                f"{page.markdown}\n"
            )

        return "\n---\n\n".join(context_parts)

    def _build_context_from_chunks(self, chunks: List[Chunk]) -> str:
        """Build context from retrieved chunks with source attribution"""
        context_parts = []

        for idx, chunk in enumerate(chunks, 1):
            metadata = chunk.metadata
            source_type = metadata.source_type.upper()
            context_parts.append(
                f"[CHUNK {idx} - {source_type}]\n"
                f"Source: {metadata.page_title}\n"
                f"URL: {metadata.page_url}\n"
                f"Section: Part {metadata.chunk_index + 1} of {metadata.total_chunks}\n\n"
                f"{chunk.text}\n"
            )

        return "\n---\n\n".join(context_parts)

    def _extract_citations_from_chunks(self, chunks: List[Chunk]) -> List[Citation]:
        """Extract unique citations from chunks"""
        seen_urls = set()
        citations = []

        for chunk in chunks:
            metadata = chunk.metadata
            if metadata.page_url not in seen_urls:
                citations.append(
                    Citation(
                        page_title=metadata.page_title,
                        page_url=metadata.page_url,
                        source_type=metadata.source_type
                    )
                )
                seen_urls.add(metadata.page_url)

        return citations

    def _parse_suggestions(self, response_text: str) -> Optional[List[PageSuggestion]]:
        """
        Parse page suggestions from LLM response

        Looks for patterns like:
        💡 **Missing Information:**
        - [Topic] - [reason]
        Keywords: keyword1, keyword2
        """
        # Check if response contains suggestions section
        if "💡" not in response_text and "Missing Information" not in response_text:
            return None

        suggestions = []

        # Extract bullet points after Missing Information
        # Pattern: - [Topic] - [reason]
        pattern = r'-\s+([^-]+?)\s+-\s+(.+?)(?=\n|$)'
        matches = re.findall(pattern, response_text, re.MULTILINE)

        if not matches:
            return None

        # Extract keywords
        keywords = []
        keyword_match = re.search(r'Keywords(?:\s+to\s+search)?:\s*(.+?)(?:\n|$)', response_text, re.IGNORECASE)
        if keyword_match:
            keywords_str = keyword_match.group(1)
            keywords = [k.strip() for k in re.split(r'[,;]', keywords_str) if k.strip()]

        # Create suggestions
        for topic, reason in matches:
            topic = topic.strip()
            reason = reason.strip()

            # Determine confidence based on specificity
            if any(kw in topic.lower() or kw in reason.lower() for kw in ["version", "update", "specific"]):
                confidence = "high"
            elif len(reason) > 30:
                confidence = "medium"
            else:
                confidence = "low"

            suggestions.append(
                PageSuggestion(
                    reason=f"{topic}: {reason}",
                    keywords=keywords if keywords else [topic],
                    confidence=confidence
                )
            )

        return suggestions if suggestions else None

    def _flatten_message_text(self, value: Any) -> list[str]:
        if value is None:
            return []

        if isinstance(value, str):
            normalized = value.strip()
            return [normalized] if normalized else []

        if isinstance(value, list):
            parts: list[str] = []
            for item in value:
                parts.extend(self._flatten_message_text(item))
            return parts

        if isinstance(value, dict):
            parts: list[str] = []
            for key in ("text", "output_text", "content"):
                if key in value:
                    parts.extend(self._flatten_message_text(value.get(key)))

            # Some providers surface reasoning summaries as nested blocks.
            if "summary" in value:
                parts.extend(self._flatten_message_text(value.get("summary")))

            refusal = str(value.get("refusal") or "").strip()
            if refusal:
                parts.append(refusal)

            return parts

        return []

    def _extract_response_text(self, response: Any) -> str:
        content_parts = self._flatten_message_text(getattr(response, "content", None))
        if content_parts:
            return "\n\n".join(part for part in content_parts if part).strip()

        additional_kwargs = getattr(response, "additional_kwargs", {}) or {}
        additional_parts = self._flatten_message_text(additional_kwargs)
        if additional_parts:
            return "\n\n".join(part for part in additional_parts if part).strip()

        response_metadata = getattr(response, "response_metadata", {}) or {}
        metadata_parts = self._flatten_message_text(response_metadata)
        if metadata_parts:
            return "\n\n".join(part for part in metadata_parts if part).strip()

        logger.warning(
            "LLM returned empty text content provider=%s model=%s content_type=%s additional_kwargs_keys=%s response_metadata_keys=%s",
            self.settings.llm_provider,
            self.get_model_name(),
            type(getattr(response, "content", None)).__name__,
            sorted(additional_kwargs.keys()) if isinstance(additional_kwargs, dict) else [],
            sorted(response_metadata.keys()) if isinstance(response_metadata, dict) else [],
        )
        return ""

    def _ensure_non_empty_response_text(
        self,
        response: Any,
        fallback_message: str
    ) -> str:
        response_text = self._extract_response_text(response)
        if response_text:
            return response_text

        logger.warning(
            "Using fallback text because the provider response body was empty provider=%s model=%s",
            self.settings.llm_provider,
            self.get_model_name(),
        )
        return fallback_message

    def _extract_citations(self, pages: List[PageContent]) -> List[Citation]:
        """Extract citations from source pages"""
        return [
            Citation(page_title=page.title, page_url=page.url, source_type=page.source_type)
            for page in pages
        ]

    def _extract_mermaid_code(self, response_text: str) -> Optional[str]:
        """Extract Mermaid code block from model response"""
        mermaid_match = re.search(r"```mermaid\s*([\s\S]+?)```", response_text, re.IGNORECASE)
        if mermaid_match:
            return mermaid_match.group(1).strip()

        lines = [line.rstrip() for line in response_text.splitlines() if line.strip()]
        if not lines:
            return None

        first_line = lines[0].strip()
        if self._infer_diagram_type_from_code(first_line) != "unknown":
            return "\n".join(lines).strip()

        return None

    def _infer_diagram_type_from_code(self, mermaid_code: str) -> str:
        """Infer Mermaid diagram type from the first meaningful line"""
        if not mermaid_code or not mermaid_code.strip():
            return "unknown"

        first_line = mermaid_code.strip().splitlines()[0].strip()

        diagram_prefixes = {
            "sequenceDiagram": "sequenceDiagram",
            "flowchart": "flowchart",
            "graph": "graph",
            "classDiagram": "classDiagram",
            "stateDiagram": "stateDiagram",
            "erDiagram": "erDiagram",
            "journey": "journey",
            "gantt": "gantt",
            "pie": "pie",
            "mindmap": "mindmap",
            "timeline": "timeline"
        }

        for prefix, diagram_type in diagram_prefixes.items():
            if first_line.startswith(prefix):
                return diagram_type

        return "unknown"

    def _validate_mermaid_syntax(self, mermaid_code: str) -> bool:
        """Perform lightweight Mermaid validation without an external parser"""
        if not mermaid_code or not mermaid_code.strip():
            return False

        diagram_type = self._infer_diagram_type_from_code(mermaid_code)
        if diagram_type == "unknown":
            return False

        lines = [line.strip() for line in mermaid_code.splitlines() if line.strip()]
        if len(lines) < 2:
            return False

        # Basic sanity checks to catch obviously broken outputs
        if any(line.startswith("```") for line in lines):
            return False

        return True

    def _extract_diagram_summary(self, response_text: str) -> str:
        """Extract prose summary from model response, excluding Mermaid block"""
        summary = re.sub(r"```mermaid\s*[\s\S]+?```", "", response_text, flags=re.IGNORECASE)
        summary = re.sub(r"\n{3,}", "\n\n", summary).strip()

        return summary or "Generated diagram based on the most relevant retrieved context."

    def _extract_json_object(self, response_text: str) -> Optional[dict]:
        """Extract a JSON object from model output."""
        fenced_match = re.search(r"```json\s*([\s\S]+?)```", response_text, re.IGNORECASE)
        candidate = fenced_match.group(1).strip() if fenced_match else response_text.strip()

        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass

        object_match = re.search(r"\{[\s\S]*\}", response_text)
        if not object_match:
            return None

        try:
            parsed = json.loads(object_match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None

    def _parse_critique_response(
        self,
        response_text: str,
        fallback_source_title: Optional[str] = None
    ) -> Tuple[str, List[CritiqueIssue]]:
        """Parse structured critique JSON with graceful fallback."""
        parsed = self._extract_json_object(response_text)
        if not parsed:
            return response_text.strip(), []

        summary = str(parsed.get("summary") or "").strip() or "Review completed."
        raw_issues = parsed.get("issues")

        if not isinstance(raw_issues, list):
            return summary, []

        issues: List[CritiqueIssue] = []

        for item in raw_issues[:5]:
            if not isinstance(item, dict):
                continue

            severity = str(item.get("severity") or "medium").strip().lower()
            if severity not in {"high", "medium", "low"}:
                severity = "medium"

            title = str(item.get("title") or "").strip()
            risk = str(item.get("risk") or "").strip()
            suggestion = str(item.get("suggestion") or "").strip()

            if not title or not risk or not suggestion:
                continue

            issues.append(
                CritiqueIssue(
                    title=title,
                    severity=severity,
                    category=str(item.get("category") or "other").strip() or "other",
                    evidence=str(item.get("evidence") or "").strip() or "Not explicitly quoted.",
                    risk=risk,
                    suggestion=suggestion,
                    source_title=str(item.get("source_title") or fallback_source_title or "").strip() or None,
                )
            )

        return summary, issues

    async def _ainvoke_with_timeout(self, messages, task_kind: str = "answer"):
        model, model_name, provider_name, model_slot, fallback_used, routing_note = self._get_model_for_task(task_kind)
        timeout_seconds = self._timeout_seconds_for_task(task_kind)
        self._last_task_kind = self._normalize_task_kind(task_kind)
        self._last_provider_name = provider_name
        self._last_provider_mode = self.settings.llm_provider
        self._last_model_name = model_name
        self._last_model_slot = model_slot
        self._last_model_fallback_used = fallback_used
        self._last_routing_note = routing_note
        self._last_generation_metrics = {}
        start_time = time.perf_counter()

        try:
            response = await asyncio.wait_for(
                model.ainvoke(messages),
                timeout=timeout_seconds
            )
            wall_clock_ms = round((time.perf_counter() - start_time) * 1000)
            self._last_generation_metrics = self._extract_generation_metrics(response, wall_clock_ms)
            return response
        except asyncio.TimeoutError as exc:
            logger.error(
                "LLM request timed out provider=%s model=%s task=%s timeout_seconds=%s",
                provider_name,
                model_name,
                self._last_task_kind,
                timeout_seconds,
            )
            raise TimeoutError(
                f"LLM request timed out after {timeout_seconds} seconds"
            ) from exc
        except Exception:
            logger.exception(
                "LLM invocation failed provider=%s model=%s task=%s",
                provider_name,
                model_name,
                self._last_task_kind,
            )
            raise

    def _build_prefilter_prompt(
        self,
        chunks: List[Chunk],
        user_question: str,
        max_keep: int,
        task_kind: str,
        planner_mode: str | None = None,
    ) -> str:
        max_chars = max(120, self.settings.hybrid_prefilter_max_chars_per_chunk)
        excerpts: list[str] = []
        compare_mode = bool(planner_mode and planner_mode.startswith("compare"))
        critique_mode = task_kind == "critique"
        diagram_mode = task_kind == "diagram"

        for idx, chunk in enumerate(chunks, 1):
            preview = re.sub(r"\s+", " ", chunk.text).strip()
            preview = preview[:max_chars]
            excerpts.append(
                f"[{idx}] source={chunk.metadata.page_title} "
                f"type={chunk.metadata.source_type} "
                f"section={chunk.metadata.chunk_index + 1}/{chunk.metadata.total_chunks} "
                f"text={preview}"
            )

        task_guidance = [
            "Prefer the minimum set of excerpts that preserves answer quality.",
            "Avoid keeping many excerpts from the same page unless they are clearly necessary.",
        ]

        if compare_mode:
            task_guidance.extend(
                [
                    "This is a compare task.",
                    "Keep evidence from at least 2 different pages when available.",
                    "Preserve contrasting claims, different values, or different assumptions across sources.",
                    "Do not collapse the context to only one side of the comparison.",
                ]
            )
        elif critique_mode:
            task_guidance.extend(
                [
                    "This is a critique task.",
                    "Prioritize excerpts that mention edge cases, failure handling, validation, retries, timeouts, security, missing requirements, or risky assumptions.",
                    "Keep the main requirement/spec excerpt plus at least one supporting or challenging excerpt from another page when available.",
                    "Do not drop negative-path or operational evidence if it changes the critique.",
                ]
            )
        elif diagram_mode:
            task_guidance.extend(
                [
                    "This is a diagram task.",
                    "Prefer excerpts that preserve actors, sequence, dependencies, and named system boundaries.",
                ]
            )

        return (
            "You are selecting the minimum set of excerpts needed for a later answer.\n"
            "Return JSON only in this shape:\n"
            '{"keep_chunk_ids":[1,2],"reason":"short reason"}\n'
            f"Task kind: {task_kind}\n"
            f"Planner mode: {planner_mode or 'unknown'}\n"
            f"Question: {user_question}\n"
            f"Keep at most {max_keep} chunks.\n"
            + "\n".join(f"- {line}" for line in task_guidance)
            + "\n\n"
            "Candidate excerpts:\n"
            + "\n".join(excerpts)
        )

    def _augment_prefilter_selection_for_diversity(
        self,
        chunks: List[Chunk],
        keep_ids: list[int],
        max_keep: int,
        task_kind: str,
        planner_mode: str | None = None,
    ) -> tuple[list[int], str | None]:
        compare_mode = bool(planner_mode and planner_mode.startswith("compare"))
        critique_mode = task_kind == "critique"

        if not compare_mode and not critique_mode:
            return keep_ids, None

        if len(chunks) <= 1:
            return keep_ids, None

        selected_ids = list(keep_ids)
        selected_pages = {chunks[idx - 1].metadata.page_url for idx in selected_ids if 1 <= idx <= len(chunks)}
        available_pages = {chunk.metadata.page_url for chunk in chunks}

        if len(available_pages) <= 1 or len(selected_pages) >= 2:
            return selected_ids, None

        candidate_id = None
        selected_page = next(iter(selected_pages), None)
        for idx, chunk in enumerate(chunks, 1):
            if chunk.metadata.page_url != selected_page:
                candidate_id = idx
                break

        if candidate_id is None:
            return selected_ids, None

        if len(selected_ids) < max_keep:
            selected_ids.append(candidate_id)
        else:
            replacement_index = None
            for list_index in range(len(selected_ids) - 1, -1, -1):
                existing_id = selected_ids[list_index]
                if chunks[existing_id - 1].metadata.page_url == selected_page:
                    replacement_index = list_index
                    break

            if replacement_index is None:
                return selected_ids, None
            selected_ids[replacement_index] = candidate_id

        adjustment_reason = (
            "Added a second page so the comparison keeps both sides of the evidence."
            if compare_mode
            else "Added a second page so critique keeps both requirement context and supporting risk evidence."
        )
        deduped_ids: list[int] = []
        seen_ids: set[int] = set()
        for chunk_id in selected_ids:
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)
            deduped_ids.append(chunk_id)

        return deduped_ids[:max_keep], adjustment_reason

    def _parse_prefilter_response(
        self,
        response_text: str,
        chunk_count: int,
        max_keep: int
    ) -> tuple[list[int], str]:
        parsed = self._extract_json_object(response_text)
        if not parsed:
            return [], "Local prefilter did not return parseable JSON."

        raw_ids = parsed.get("keep_chunk_ids")
        reason = str(parsed.get("reason") or "").strip()
        if not isinstance(raw_ids, list):
            return [], reason or "Local prefilter JSON was missing keep_chunk_ids."

        keep_ids: list[int] = []
        seen: set[int] = set()
        for raw_id in raw_ids:
            try:
                chunk_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            if chunk_id < 1 or chunk_id > chunk_count or chunk_id in seen:
                continue
            seen.add(chunk_id)
            keep_ids.append(chunk_id)
            if len(keep_ids) >= max_keep:
                break

        return keep_ids, reason

    async def prefilter_chunks(
        self,
        chunks: List[Chunk],
        user_question: str,
        task_kind: str = "answer",
        planner_mode: str | None = None,
    ) -> tuple[List[Chunk], dict[str, Any]]:
        if not self.settings.hybrid_mode_enabled():
            return chunks, {"hybrid_prefilter_applied": False}

        descriptor = self.models.get("prefilter")
        if (
            not self.settings.hybrid_prefilter_enabled
            or descriptor is None
            or not descriptor.get("configured")
            or len(chunks) <= 1
        ):
            return chunks, {
                "hybrid_prefilter_applied": False,
                "hybrid_prefilter_reason": "Hybrid prefilter disabled or not needed.",
            }

        max_keep = max(1, min(self.settings.hybrid_prefilter_max_chunks, len(chunks)))
        if len(chunks) <= max_keep:
            return chunks, {
                "hybrid_prefilter_applied": False,
                "hybrid_prefilter_reason": "Retrieved chunk set already fits the hybrid target.",
            }

        prompt = self._build_prefilter_prompt(chunks, user_question, max_keep, task_kind, planner_mode)
        messages = [
            SystemMessage(content="Return strict JSON only."),
            HumanMessage(content=prompt),
        ]
        started_at = time.perf_counter()

        try:
            response = await asyncio.wait_for(
                descriptor["client"].ainvoke(messages),
                timeout=self.settings.hybrid_prefilter_timeout_seconds,
            )
            response_text = self._ensure_non_empty_response_text(
                response,
                "Hybrid prefilter returned no content."
            )
            keep_ids, reason = self._parse_prefilter_response(response_text, len(chunks), max_keep)
            keep_ids, diversity_reason = self._augment_prefilter_selection_for_diversity(
                chunks,
                keep_ids,
                max_keep,
                task_kind,
                planner_mode,
            )
            if diversity_reason:
                reason = f"{reason} {diversity_reason}".strip() if reason else diversity_reason
            if not keep_ids:
                return chunks, {
                    "hybrid_prefilter_applied": False,
                    "hybrid_prefilter_fallback_used": True,
                    "hybrid_prefilter_provider": "ollama",
                    "hybrid_prefilter_model": descriptor["model_name"],
                    "hybrid_prefilter_input_chunks": len(chunks),
                    "hybrid_prefilter_output_chunks": len(chunks),
                    "hybrid_prefilter_latency_ms": round((time.perf_counter() - started_at) * 1000),
                    "hybrid_prefilter_reason": reason or "Hybrid prefilter returned no usable keep list.",
                }

            kept_set = set(keep_ids)
            selected_chunks = [chunk for idx, chunk in enumerate(chunks, 1) if idx in kept_set]
            if not selected_chunks:
                selected_chunks = chunks[:max_keep]

            return selected_chunks, {
                "hybrid_prefilter_applied": True,
                "hybrid_prefilter_provider": "ollama",
                "hybrid_prefilter_model": descriptor["model_name"],
                "hybrid_prefilter_input_chunks": len(chunks),
                "hybrid_prefilter_output_chunks": len(selected_chunks),
                "hybrid_prefilter_reduced_chunks": max(0, len(chunks) - len(selected_chunks)),
                "hybrid_prefilter_latency_ms": round((time.perf_counter() - started_at) * 1000),
                "hybrid_prefilter_reason": reason or "Local Ollama prefilter reduced the context basket before the cloud answer.",
            }
        except asyncio.TimeoutError:
            logger.warning(
                "Hybrid prefilter timed out provider=ollama model=%s timeout_seconds=%s",
                descriptor["model_name"],
                self.settings.hybrid_prefilter_timeout_seconds,
            )
            return chunks, {
                "hybrid_prefilter_applied": False,
                "hybrid_prefilter_fallback_used": True,
                "hybrid_prefilter_provider": "ollama",
                "hybrid_prefilter_model": descriptor["model_name"],
                "hybrid_prefilter_input_chunks": len(chunks),
                "hybrid_prefilter_output_chunks": len(chunks),
                "hybrid_prefilter_latency_ms": round((time.perf_counter() - started_at) * 1000),
                "hybrid_prefilter_reason": "Local prefilter timed out, so the cloud model received the full retrieved context.",
            }
        except Exception as exc:
            logger.warning(
                "Hybrid prefilter failed provider=ollama model=%s error=%s",
                descriptor["model_name"],
                str(exc),
            )
            return chunks, {
                "hybrid_prefilter_applied": False,
                "hybrid_prefilter_fallback_used": True,
                "hybrid_prefilter_provider": "ollama",
                "hybrid_prefilter_model": descriptor["model_name"],
                "hybrid_prefilter_input_chunks": len(chunks),
                "hybrid_prefilter_output_chunks": len(chunks),
                "hybrid_prefilter_latency_ms": round((time.perf_counter() - started_at) * 1000),
                "hybrid_prefilter_reason": "Local prefilter failed, so the cloud model received the full retrieved context.",
            }

    async def summarize(
        self, pages: List[PageContent], user_question: str = None
    ) -> Tuple[str, List[Citation], dict]:
        """
        Summarize multiple pages with optional user question

        Args:
            pages: List of page contents to summarize
            user_question: Optional specific question to answer

        Returns:
            Tuple of (summary, citations, token_usage)
        """
        system_prompt = self._build_system_prompt()
        context = self._build_context(pages)

        # Build user prompt
        if user_question:
            user_prompt = (
                f"Based on the following documents, please answer this question:\n\n"
                f"Question: {user_question}\n\n"
                f"---\n\n"
                f"Documents:\n\n{context}"
            )
        else:
            user_prompt = f"Please summarize the following documents:\n\n{context}"

        # Prepare messages
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

        # Invoke LLM
        response = await self._ainvoke_with_timeout(messages, task_kind="answer")

        # Extract token usage
        token_usage_raw = response.response_metadata.get("token_usage", {})
        token_usage = {
            "input_tokens": token_usage_raw.get("prompt_tokens", 0),
            "output_tokens": token_usage_raw.get("completion_tokens", 0),
            "total_tokens": token_usage_raw.get("total_tokens", 0),
        }
        token_usage.update(self._build_llm_usage_metadata())

        response_text = self._ensure_non_empty_response_text(
            response,
            "The model returned an empty response. Please retry this request."
        )

        # Extract citations
        citations = self._extract_citations(pages)

        return response_text, citations, token_usage

    async def summarize_from_chunks(
        self, chunks: List[Chunk], user_question: str
    ) -> Tuple[str, List[Citation], dict, Optional[List[PageSuggestion]]]:
        """
        Summarize based on retrieved chunks (Phase 2 RAG)

        Args:
            chunks: Retrieved chunks from vector search
            user_question: User's question

        Returns:
            Tuple of (summary, citations, token_usage, suggestions)
        """
        diagram_mode = self._is_diagram_request(user_question)
        system_prompt = (
            self._build_diagram_system_prompt()
            if diagram_mode
            else self._build_system_prompt()
        )
        context = self._build_context_from_chunks(chunks)

        # Build user prompt
        if diagram_mode:
            user_prompt = (
                f"Based on the following relevant excerpts, create the best Mermaid diagram for this request:\n\n"
                f"Question: {user_question}\n\n"
                f"---\n\n"
                f"Relevant Information:\n\n{context}"
            )
        else:
            user_prompt = (
                f"Based on the following relevant excerpts, please answer this question:\n\n"
                f"Question: {user_question}\n\n"
                f"---\n\n"
                f"Relevant Information:\n\n{context}"
            )

        # Prepare messages
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

        # Invoke LLM
        response = await self._ainvoke_with_timeout(messages, task_kind="answer")

        # Extract token usage
        token_usage_raw = response.response_metadata.get("token_usage", {})
        token_usage = {
            "input_tokens": token_usage_raw.get("prompt_tokens", 0),
            "output_tokens": token_usage_raw.get("completion_tokens", 0),
            "total_tokens": token_usage_raw.get("total_tokens", 0),
        }
        token_usage.update(self._build_llm_usage_metadata())

        response_text = self._ensure_non_empty_response_text(
            response,
            "The model returned an empty answer. Please retry this request."
        )

        # Extract unique citations from chunks
        citations = self._extract_citations_from_chunks(chunks)

        # Parse suggestions for missing pages
        suggestions = self._parse_suggestions(response_text)

        return response_text, citations, token_usage, suggestions

    async def generate_diagram_from_chunks(
        self,
        chunks: List[Chunk],
        user_question: str,
        diagram_type: Optional[str] = None
    ) -> Tuple[str, str, bool, str, List[Citation], dict]:
        """
        Generate a Mermaid diagram from retrieved chunks

        Returns:
            Tuple of (summary, mermaid_code, is_valid, diagram_type, citations, token_usage)
        """
        system_prompt = self._build_diagram_system_prompt(diagram_type)
        context = self._build_context_from_chunks(chunks)

        if diagram_type:
            user_prompt = (
                f"Based on the following relevant excerpts, create a Mermaid {diagram_type} for this request:\n\n"
                f"Question: {user_question}\n\n"
                f"---\n\n"
                f"Relevant Information:\n\n{context}"
            )
        else:
            user_prompt = (
                f"Based on the following relevant excerpts, create the best Mermaid diagram for this request:\n\n"
                f"Question: {user_question}\n\n"
                f"---\n\n"
                f"Relevant Information:\n\n{context}"
            )

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        response = await self._ainvoke_with_timeout(messages, task_kind="diagram")

        token_usage_raw = response.response_metadata.get("token_usage", {})
        token_usage = {
            "input_tokens": token_usage_raw.get("prompt_tokens", 0),
            "output_tokens": token_usage_raw.get("completion_tokens", 0),
            "total_tokens": token_usage_raw.get("total_tokens", 0),
        }
        token_usage.update(self._build_llm_usage_metadata())

        response_text = self._ensure_non_empty_response_text(
            response,
            "The model returned no diagram explanation."
        )

        citations = self._extract_citations_from_chunks(chunks)
        mermaid_code = self._extract_mermaid_code(response_text) or ""
        detected_diagram_type = self._infer_diagram_type_from_code(mermaid_code)
        is_valid = self._validate_mermaid_syntax(mermaid_code)
        summary = self._extract_diagram_summary(response_text)

        return (
            summary,
            mermaid_code,
            is_valid,
            diagram_type or detected_diagram_type,
            citations,
            token_usage
        )

    async def critique_from_chunks(
        self,
        chunks: List[Chunk],
        user_question: Optional[str] = None
    ) -> Tuple[str, List[CritiqueIssue], List[Citation], dict]:
        """
        Critique requirements based on retrieved chunks.

        Returns:
            Tuple of (summary, issues, citations, token_usage)
        """
        system_prompt = self._build_critic_system_prompt()
        context = self._build_context_from_chunks(chunks)
        prompt_focus = user_question or "Review these documents for missing requirements, edge cases, and risks."

        user_prompt = (
            "Review the following relevant excerpts.\n\n"
            f"Review focus: {prompt_focus}\n\n"
            "---\n\n"
            f"Relevant Information:\n\n{context}"
        )

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        response = await self._ainvoke_with_timeout(messages, task_kind="critique")

        token_usage_raw = response.response_metadata.get("token_usage", {})
        token_usage = {
            "input_tokens": token_usage_raw.get("prompt_tokens", 0),
            "output_tokens": token_usage_raw.get("completion_tokens", 0),
            "total_tokens": token_usage_raw.get("total_tokens", 0),
        }
        token_usage.update(self._build_llm_usage_metadata())

        response_text = self._ensure_non_empty_response_text(
            response,
            "Review completed, but the model returned no written summary."
        )

        citations = self._extract_citations_from_chunks(chunks)
        fallback_source_title = citations[0].page_title if citations else None
        summary, issues = self._parse_critique_response(
            response_text,
            fallback_source_title=fallback_source_title
        )

        return summary, issues, citations, token_usage

    def get_model_name(self, task_kind: str | None = None) -> str:
        """Get current model identifier"""
        if task_kind is not None:
            _, model_name, _, _, _, _ = self._get_model_for_task(task_kind)
            return model_name
        return self._last_model_name
