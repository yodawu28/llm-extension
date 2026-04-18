"""
Token Counter Service - Accurate token counting using tiktoken
Ensures context window limits are respected
"""

from typing import List
import tiktoken

from config.settings import Settings
from services.chunking_service import Chunk


class TokenCounter:
    """Count tokens accurately using tiktoken encoding"""

    def __init__(self, settings: Settings):
        self.settings = settings

        # Use cl100k_base encoding (GPT-4, GPT-3.5-turbo, text-embedding-ada-002)
        # This works for both OpenAI and Anthropic models (similar tokenization)
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_text(self, text: str) -> int:
        """
        Count tokens in a text string

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        if not text:
            return 0

        return len(self.encoding.encode(text))

    def count_chunks(self, chunks: List[Chunk]) -> int:
        """
        Count total tokens across multiple chunks

        Args:
            chunks: List of chunks

        Returns:
            Total token count
        """
        return sum(self.count_text(chunk.text) for chunk in chunks)

    def select_chunks_within_budget(
        self,
        chunks_with_scores: List[tuple[Chunk, float]],
        reserve_tokens: int = 4096,
        max_chunks_per_page: int = 3,
        page_chunk_caps: dict[str, int] | None = None,
        source_type_chunk_caps: dict[str, int] | None = None,
        max_total_chunks: int | None = None,
        max_input_tokens_override: int | None = None
    ) -> List[Chunk]:
        """
        Select chunks that fit within token budget

        Strategy:
        1. Sort by similarity score (highest first)
        2. Add chunks until budget is reached
        3. Always include at least 1 chunk (highest similarity)

        Args:
            chunks_with_scores: List of (Chunk, similarity_score) tuples
            reserve_tokens: Tokens to reserve for output + overhead

        Returns:
            List of selected chunks within budget
        """
        # Calculate available budget for input
        max_input = (
            max_input_tokens_override
            if max_input_tokens_override is not None
            else self.settings.max_input_tokens - reserve_tokens
        )

        # Sort by similarity (descending)
        sorted_chunks = sorted(chunks_with_scores, key=lambda x: x[1], reverse=True)

        selected_chunks = []
        total_tokens = 0
        page_chunk_counts: dict[str, int] = {}
        source_type_counts: dict[str, int] = {}
        deferred_chunks: List[tuple[Chunk, float]] = []

        for chunk, score in sorted_chunks:
            chunk_tokens = self.count_text(chunk.text)
            page_url = chunk.metadata.page_url
            source_type = chunk.metadata.source_type
            page_cap = max(1, page_chunk_caps.get(page_url, max_chunks_per_page)) if page_chunk_caps else max_chunks_per_page
            source_type_cap = (
                max(1, source_type_chunk_caps.get(source_type, len(sorted_chunks)))
                if source_type_chunk_caps
                else len(sorted_chunks)
            )

            # Always include first chunk (most relevant)
            if len(selected_chunks) == 0:
                selected_chunks.append(chunk)
                total_tokens += chunk_tokens
                page_chunk_counts[page_url] = 1
                source_type_counts[source_type] = source_type_counts.get(source_type, 0) + 1
                continue

            if max_total_chunks is not None and len(selected_chunks) >= max_total_chunks:
                deferred_chunks.append((chunk, score))
                continue

            if page_chunk_counts.get(page_url, 0) >= 1:
                deferred_chunks.append((chunk, score))
                continue

            if source_type_counts.get(source_type, 0) >= source_type_cap:
                deferred_chunks.append((chunk, score))
                continue

            # Check if adding this chunk exceeds budget
            if total_tokens + chunk_tokens <= max_input:
                selected_chunks.append(chunk)
                total_tokens += chunk_tokens
                page_chunk_counts[page_url] = page_chunk_counts.get(page_url, 0) + 1
                source_type_counts[source_type] = source_type_counts.get(source_type, 0) + 1
            else:
                deferred_chunks.append((chunk, score))

        for chunk, score in deferred_chunks:
            page_url = chunk.metadata.page_url
            source_type = chunk.metadata.source_type

            page_cap = max(1, page_chunk_caps.get(page_url, max_chunks_per_page)) if page_chunk_caps else max_chunks_per_page
            source_type_cap = (
                max(1, source_type_chunk_caps.get(source_type, len(sorted_chunks)))
                if source_type_chunk_caps
                else len(sorted_chunks)
            )

            if page_chunk_counts.get(page_url, 0) >= page_cap:
                continue

            if source_type_counts.get(source_type, 0) >= source_type_cap:
                continue

            if max_total_chunks is not None and len(selected_chunks) >= max_total_chunks:
                continue

            chunk_tokens = self.count_text(chunk.text)

            if total_tokens + chunk_tokens <= max_input:
                selected_chunks.append(chunk)
                total_tokens += chunk_tokens
                page_chunk_counts[page_url] = page_chunk_counts.get(page_url, 0) + 1
                source_type_counts[source_type] = source_type_counts.get(source_type, 0) + 1

        return selected_chunks

    def validate_input_size(self, text: str) -> tuple[bool, int]:
        """
        Validate if text fits within max input tokens

        Args:
            text: Text to validate

        Returns:
            Tuple of (is_valid, token_count)
        """
        token_count = self.count_text(text)
        is_valid = token_count <= self.settings.max_input_tokens
        return is_valid, token_count

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> dict:
        """
        Estimate API cost based on token usage

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Dictionary with cost breakdown
        """
        # Pricing (as of 2024)
        # GPT-4o: $2.50 / 1M input, $10.00 / 1M output
        # Claude 3.5 Sonnet: $3.00 / 1M input, $15.00 / 1M output

        cost_provider = self.settings.resolved_cloud_provider()

        if cost_provider == "ollama":
            input_cost_per_1m = 0.0
            output_cost_per_1m = 0.0
        elif cost_provider == "openai":
            input_cost_per_1m = 2.50
            output_cost_per_1m = 10.00
        else:  # anthropic
            input_cost_per_1m = 3.00
            output_cost_per_1m = 15.00

        input_cost = (input_tokens / 1_000_000) * input_cost_per_1m
        output_cost = (output_tokens / 1_000_000) * output_cost_per_1m
        total_cost = input_cost + output_cost

        return {
            "input_cost": round(input_cost, 4),
            "output_cost": round(output_cost, 4),
            "total_cost": round(total_cost, 4),
            "currency": "USD"
        }
