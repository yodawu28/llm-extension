import asyncio
import json
import logging
import re
from typing import List, Optional, Tuple

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
        self.model = self._init_model()
        logger.info(
            "LLM client initialized provider=%s model=%s base_url=%s reasoning_effort=%s timeout_seconds=%s",
            self.settings.llm_provider,
            self.get_model_name(),
            self.settings.resolved_openai_base_url(),
            self.settings.openai_reasoning_effort or None,
            self.settings.request_timeout_seconds,
        )

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

    def _init_model(self):
        """Initialize LLM based on provider setting"""
        try:
            if self.settings.llm_provider == "openai":
                api_key = self.settings.resolved_openai_api_key()
                if not api_key:
                    raise ValueError("OpenAI API key or PAT token not found in settings")

                init_kwargs = dict(
                    api_key=api_key,
                    model=self.settings.openai_model,
                    base_url=self.settings.resolved_openai_base_url(),
                    max_tokens=self.settings.max_output_tokens,
                    timeout=self.settings.request_timeout_seconds,
                    max_retries=1,
                )

                if self._supports_custom_temperature(self.settings.openai_model):
                    init_kwargs["temperature"] = 0.3
                else:
                    # LangChain defaults ChatOpenAI.temperature to 0.7. Reasoning models
                    # like o4-mini only accept the provider default temperature of 1.
                    init_kwargs["temperature"] = 1

                if self.settings.openai_reasoning_effort:
                    init_kwargs["reasoning_effort"] = self.settings.openai_reasoning_effort

                return ChatOpenAI(**init_kwargs)

            if self.settings.llm_provider == "anthropic":
                if not self.settings.anthropic_api_key:
                    raise ValueError("Anthropic API key not found in settings")

                return ChatAnthropic(
                    api_key=self.settings.anthropic_api_key,
                    model=self.settings.anthropic_model,
                    temperature=0.3,
                    max_tokens=self.settings.max_output_tokens,
                    timeout=self.settings.request_timeout_seconds,
                    max_retries=1,
                )

            raise ValueError(f"Unknown LLM provider: {self.settings.llm_provider}")
        except Exception:
            logger.exception(
                "Failed to initialize LLM client provider=%s model=%s base_url=%s has_pat_token=%s has_openai_api_key=%s",
                self.settings.llm_provider,
                self.settings.openai_model if self.settings.llm_provider == "openai" else self.settings.anthropic_model,
                self.settings.resolved_openai_base_url(),
                bool(self.settings.pat_token),
                bool(self.settings.openai_api_key),
            )
            raise

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

    async def _ainvoke_with_timeout(self, messages):
        try:
            return await asyncio.wait_for(
                self.model.ainvoke(messages),
                timeout=self.settings.request_timeout_seconds
            )
        except asyncio.TimeoutError as exc:
            logger.error(
                "LLM request timed out provider=%s model=%s timeout_seconds=%s",
                self.settings.llm_provider,
                self.get_model_name(),
                self.settings.request_timeout_seconds,
            )
            raise TimeoutError(
                f"LLM request timed out after {self.settings.request_timeout_seconds} seconds"
            ) from exc
        except Exception:
            logger.exception(
                "LLM invocation failed provider=%s model=%s",
                self.settings.llm_provider,
                self.get_model_name(),
            )
            raise

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
        response = await self._ainvoke_with_timeout(messages)

        # Extract token usage
        token_usage_raw = response.response_metadata.get("token_usage", {})
        token_usage = {
            "input_tokens": token_usage_raw.get("prompt_tokens", 0),
            "output_tokens": token_usage_raw.get("completion_tokens", 0),
            "total_tokens": token_usage_raw.get("total_tokens", 0),
        }

        # Extract citations
        citations = self._extract_citations(pages)

        return response.content, citations, token_usage

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
        response = await self._ainvoke_with_timeout(messages)

        # Extract token usage
        token_usage_raw = response.response_metadata.get("token_usage", {})
        token_usage = {
            "input_tokens": token_usage_raw.get("prompt_tokens", 0),
            "output_tokens": token_usage_raw.get("completion_tokens", 0),
            "total_tokens": token_usage_raw.get("total_tokens", 0),
        }

        # Extract unique citations from chunks
        citations = self._extract_citations_from_chunks(chunks)

        # Parse suggestions for missing pages
        suggestions = self._parse_suggestions(response.content)

        return response.content, citations, token_usage, suggestions

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
        response = await self._ainvoke_with_timeout(messages)

        token_usage_raw = response.response_metadata.get("token_usage", {})
        token_usage = {
            "input_tokens": token_usage_raw.get("prompt_tokens", 0),
            "output_tokens": token_usage_raw.get("completion_tokens", 0),
            "total_tokens": token_usage_raw.get("total_tokens", 0),
        }

        citations = self._extract_citations_from_chunks(chunks)
        mermaid_code = self._extract_mermaid_code(response.content) or ""
        detected_diagram_type = self._infer_diagram_type_from_code(mermaid_code)
        is_valid = self._validate_mermaid_syntax(mermaid_code)
        summary = self._extract_diagram_summary(response.content)

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
        response = await self._ainvoke_with_timeout(messages)

        token_usage_raw = response.response_metadata.get("token_usage", {})
        token_usage = {
            "input_tokens": token_usage_raw.get("prompt_tokens", 0),
            "output_tokens": token_usage_raw.get("completion_tokens", 0),
            "total_tokens": token_usage_raw.get("total_tokens", 0),
        }

        citations = self._extract_citations_from_chunks(chunks)
        fallback_source_title = citations[0].page_title if citations else None
        summary, issues = self._parse_critique_response(
            response.content,
            fallback_source_title=fallback_source_title
        )

        return summary, issues, citations, token_usage

    def get_model_name(self) -> str:
        """Get current model identifier"""
        if self.settings.llm_provider == "openai":
            return self.settings.openai_model
        return self.settings.anthropic_model
