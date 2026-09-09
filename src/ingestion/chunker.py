# src/ingestion/chunker.py
from dataclasses import dataclass
from typing import List
import re

@dataclass
class Chunk:
    """A semantic block-oriented chunk for technical documents."""
    text: str
    metadata: dict | None = None

def is_table_of_contents(text: str) -> bool:
    """Detects and filters out Table of Contents pages or blocks."""
    lower_text = text.lower()
    has_toc_phrase = "table of contents" in lower_text or "roadmap" in lower_text
    has_page_numbers = len(re.findall(r'\d+', text)) > 10
    has_leader_dots = "---" in text or "___" in text
    return has_toc_phrase and (has_page_numbers or has_leader_dots)

def chunk_document(text: str, max_chars: int = 2000) -> List[Chunk]:
    """Split technical documents into coherent semantic chunks, preserving tables and filtering TOCs."""
    if not text or not text.strip():
        return []

    raw_blocks = split_into_semantic_blocks(text)
    chunks: List[Chunk] = []
    current_parts: List[str] = []
    current_length = 0

    def flush_current() -> None:
        nonlocal current_parts, current_length
        if current_parts:
            joined_text = "\n\n".join(current_parts).strip()
            if not is_table_of_contents(joined_text):
                chunks.append(
                    Chunk(
                        text=joined_text,
                        metadata={"type": "block"},
                    )
                )
            current_parts = []
            current_length = 0

    for block in raw_blocks:
        block_text = block.strip()
        if not block_text:
            continue

        if is_structural_block(block_text):
            flush_current()
            if not is_table_of_contents(block_text):
                block_type = "table" if "|" in block_text else "structural"
                chunks.append(Chunk(text=block_text, metadata={"type": block_type}))
            continue

        if current_length + len(block_text) + 2 > max_chars and current_parts:
            flush_current()

        current_parts.append(block_text)
        current_length += len(block_text) + 2

    flush_current()
    return chunks

def split_into_semantic_blocks(text: str) -> List[str]:
    """Group lines into logical blocks, fixing PDF hard line wraps and keeping tables/lists intact."""
    lines = text.splitlines()
    blocks: List[str] = []
    current_block: List[str] = []
    
    in_code_fence = False
    in_table = False

    def commit() -> None:
        if current_block:
            blocks.append("\n".join(current_block).strip())
            current_block.clear()

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("```") or stripped.startswith("~~~"):
            if not in_code_fence:
                commit()
                in_code_fence = True
                current_block.append(line)
            else:
                current_block.append(line)
                commit()
                in_code_fence = False
            continue

        if in_code_fence:
            current_block.append(line)
            continue

        if stripped.startswith("#"):
            commit()
            blocks.append(line)
            continue

        is_table_row = stripped.startswith("|") and stripped.endswith("|")
        if is_table_row:
            if not in_table:
                commit()
                in_table = True
            current_block.append(line)
            continue
        elif in_table and not is_table_row:
            commit()
            in_table = False

        if not stripped:
            commit()
            continue

        is_list_item = re.match(r"^(\*|-|\d+\.)\s+", stripped)
        if is_list_item and current_block:
            commit()

        current_block.append(line)

    commit()
    return blocks

def is_structural_block(text: str) -> bool:
    """Identify blocks that must stand alone as independent chunks."""
    stripped = text.strip()
    return (
        stripped.startswith("#") 
        or stripped.startswith("```") 
        or stripped.startswith("~~~")
        or ("|" in stripped and ("-|-" in stripped or "---|--" in stripped or "|---" in stripped))
    )