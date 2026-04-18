from typing import Any, List, Optional
from urllib.parse import urlparse

from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator

MAX_PAGES_PER_REQUEST = 20


def _clean_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _clean_metadata(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def infer_source_type(url: str) -> str:
    normalized_url = _clean_string(url).lower()

    if (
        "/browse/" in normalized_url or
        "jira." in normalized_url or
        "atlassian.net/browse/" in normalized_url
    ):
        return "jira"

    if (
        "confluence" in normalized_url or
        "atlassian.net/wiki" in normalized_url or
        "/pages/viewpage.action" in normalized_url or
        "wiki." in normalized_url
    ):
        return "confluence"

    return "generic"


def _fallback_title(url: str) -> str:
    normalized_url = _clean_string(url)
    if not normalized_url:
        return "Untitled page"

    parsed = urlparse(normalized_url)
    if parsed.path and parsed.path != "/":
        tail = parsed.path.rstrip("/").split("/")[-1]
        if tail:
            return tail

    if parsed.netloc:
        return parsed.netloc

    return normalized_url


class PageContent(BaseModel):
    """Single page/issue content"""

    title: str = Field(default="", description="Page or issue title")
    url: str = Field(default="", description="Full URL")
    markdown: str = Field(default="", description="Markdown content")
    source_type: str = Field(
        default="generic",
        validation_alias=AliasChoices("source_type", "sourceType"),
        description="confluence, jira, or generic"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @model_validator(mode="before")
    @classmethod
    def normalize_input(cls, value: Any):
        if not isinstance(value, dict):
            return value

        normalized = dict(value)
        normalized["title"] = _clean_string(normalized.get("title"))
        normalized["url"] = _clean_string(normalized.get("url"))
        normalized["markdown"] = _clean_string(normalized.get("markdown"))
        normalized["metadata"] = _clean_metadata(normalized.get("metadata"))

        source_type = _clean_string(
            normalized.get("source_type", normalized.get("sourceType"))
        ).lower()
        normalized["source_type"] = source_type or infer_source_type(normalized["url"])

        if not normalized["title"]:
            normalized["title"] = _fallback_title(normalized["url"])

        return normalized

    @field_validator("source_type")
    @classmethod
    def normalize_source_type(cls, value: str) -> str:
        normalized = _clean_string(value).lower()
        return normalized if normalized in {"confluence", "jira", "generic"} else "generic"

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "title": "Feature X Documentation",
                "url": "https://example.com/feature-x",
                "markdown": "# Feature X\n\nThis feature...",
                "source_type": "generic",
                "metadata": {"word_count": 450, "token_estimate": 1200},
            }
        }
    }


class SummarizeRequest(BaseModel):
    """Request for summarization"""

    pages: List[PageContent] = Field(default_factory=list, max_length=MAX_PAGES_PER_REQUEST)
    user_question: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("user_question", "userQuestion"),
        description="Optional specific question"
    )

    @field_validator("pages", mode="before")
    @classmethod
    def normalize_pages(cls, value: Any):
        return value if isinstance(value, list) else []

    @field_validator("user_question", mode="before")
    @classmethod
    def normalize_user_question(cls, value: Any):
        normalized = _clean_string(value)
        return normalized or None

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "pages": [
                    {
                        "title": "React Documentation",
                        "url": "https://react.dev/",
                        "markdown": "# React\n\nA JavaScript library...",
                        "source_type": "generic",
                        "metadata": {},
                    }
                ],
                "user_question": "What are the main features of React?",
            }
        }
    }


class DiagramRequest(BaseModel):
    """Request for Mermaid diagram generation"""

    pages: List[PageContent] = Field(default_factory=list, max_length=MAX_PAGES_PER_REQUEST)
    user_question: str = Field(
        default="",
        validation_alias=AliasChoices("user_question", "userQuestion"),
        description="Question describing the desired diagram"
    )
    diagram_type: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("diagram_type", "diagramType"),
        description="Optional preferred Mermaid diagram type, e.g. sequenceDiagram or flowchart"
    )

    @field_validator("pages", mode="before")
    @classmethod
    def normalize_pages(cls, value: Any):
        return value if isinstance(value, list) else []

    @field_validator("user_question", mode="before")
    @classmethod
    def normalize_user_question(cls, value: Any):
        return _clean_string(value)

    @field_validator("diagram_type", mode="before")
    @classmethod
    def normalize_diagram_type(cls, value: Any):
        normalized = _clean_string(value)
        return normalized or None

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "pages": [
                    {
                        "title": "Authentication PRD",
                        "url": "https://example.com/prd/authentication",
                        "markdown": "# Authentication\n\nUser logs in with email and password...",
                        "source_type": "confluence",
                        "metadata": {},
                    }
                ],
                "user_question": "Show the authentication flow as a Mermaid sequence diagram",
                "diagram_type": "sequenceDiagram",
            }
        }
    }


class Citation(BaseModel):
    """Source citation"""

    page_title: str
    page_url: str
    source_type: str


class PageSuggestion(BaseModel):
    """Suggested page that might contain relevant information"""

    reason: str = Field(..., description="Why this page might be relevant")
    keywords: List[str] = Field(..., description="Keywords to search for")
    confidence: str = Field(..., description="high, medium, or low")


class CritiqueIssue(BaseModel):
    """Structured issue found during requirement critique"""

    title: str = Field(..., description="Short issue title")
    severity: str = Field(..., description="high, medium, or low")
    category: str = Field(..., description="Issue category, e.g. reliability or security")
    evidence: str = Field(..., description="Exact quote or paraphrased evidence from the docs")
    risk: str = Field(..., description="Why the issue matters")
    suggestion: str = Field(..., description="Suggested mitigation or follow-up")
    source_title: Optional[str] = Field(
        default=None,
        description="Best matching source page title for this issue"
    )


class SourceConflict(BaseModel):
    """High-confidence conflict detected across retrieved sources"""

    topic: str = Field(..., description="Short conflict topic label")
    summary: str = Field(..., description="Short summary of the contradiction")
    source_a_title: str = Field(..., description="First source title")
    source_a_evidence: str = Field(..., description="Supporting evidence from the first source")
    source_b_title: str = Field(..., description="Second source title")
    source_b_evidence: str = Field(..., description="Supporting evidence from the second source")


class SummarizeResponse(BaseModel):
    """Response with summary and citations"""

    summary: str = Field(..., description="Generated summary")
    citations: List[Citation] = Field(default_factory=list, description="Sources used")
    source_conflicts: Optional[List[SourceConflict]] = Field(
        default=None,
        description="Potential contradictions detected across sources"
    )
    token_usage: dict = Field(..., description="Token usage stats")
    model_used: str = Field(..., description="LLM model identifier")
    suggestions: Optional[List[PageSuggestion]] = Field(
        default=None,
        description="Suggested pages to pin for better answers"
    )


class CritiqueResponse(BaseModel):
    """Response with review findings and citations"""

    summary: str = Field(..., description="Overall review summary")
    issues: List[CritiqueIssue] = Field(default_factory=list, description="Structured critique issues")
    citations: List[Citation] = Field(default_factory=list, description="Sources used")
    source_conflicts: Optional[List[SourceConflict]] = Field(
        default=None,
        description="Potential contradictions detected across sources"
    )
    token_usage: dict = Field(..., description="Token usage stats")
    model_used: str = Field(..., description="LLM model identifier")


class DiagramResponse(BaseModel):
    """Response with Mermaid diagram output"""

    summary: str = Field(..., description="Short explanation of the diagram")
    mermaid_code: str = Field(..., description="Validated Mermaid code")
    is_valid: bool = Field(..., description="Whether Mermaid validation passed")
    diagram_type: str = Field(..., description="Detected or requested Mermaid diagram type")
    citations: List[Citation] = Field(default_factory=list, description="Sources used")
    source_conflicts: Optional[List[SourceConflict]] = Field(
        default=None,
        description="Potential contradictions detected across sources"
    )
    token_usage: dict = Field(..., description="Token usage stats")
    model_used: str = Field(..., description="LLM model identifier")


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    app_name: str
    llm_provider: str
    model: str
