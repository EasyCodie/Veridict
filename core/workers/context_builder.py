"""Context Builder - Minimal context injection for MAKER workers.

Per MAKER: "Agents receive minimal context to prevent hallucination."

This module sanitizes and limits context before passing to workers,
removing noise that could confuse the LLM.
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ContextConfig:
    """Configuration for context building."""
    max_text_length: int = 2000  # Max chars (~500 tokens)
    max_title_length: int = 100
    strip_page_numbers: bool = True
    strip_cross_references: bool = True
    strip_headers_footers: bool = True
    normalize_whitespace: bool = True


class ContextBuilder:
    """Builds minimal context for workers.
    
    Sanitizes and limits clause text to prevent:
    1. Token overflow
    2. Hallucination from cross-references
    3. Confusion from formatting artifacts
    """
    
    # Patterns to strip
    PAGE_NUMBER_PATTERNS = [
        r'Page\s*\d+\s*(of\s*\d+)?',
        r'\b\d+\s*$',  # Trailing page numbers
        r'^\s*\d+\s*\n',  # Leading page numbers
    ]
    
    CROSS_REF_PATTERNS = [
        r'[Ss]ee\s+[Ss]ection\s+[\d\.]+',
        r'[Ss]ee\s+[Aa]rticle\s+[\d\.]+',
        r'[Aa]s\s+defined\s+in\s+[Ss]ection\s+[\d\.]+',
        r'[Pp]ursuant\s+to\s+[Ss]ection\s+[\d\.]+',
        r'\([Ss]ee\s+.*?\)',
    ]
    
    HEADER_FOOTER_PATTERNS = [
        r'^.*CONFIDENTIAL.*$',
        r'^.*DRAFT.*$',
        r'^\s*-+\s*$',  # Divider lines
    ]
    
    def __init__(self, config: Optional[ContextConfig] = None):
        """Initialize with optional config."""
        self.config = config or ContextConfig()
        
        # Pre-compile patterns
        self._page_patterns = [re.compile(p, re.MULTILINE | re.IGNORECASE) 
                              for p in self.PAGE_NUMBER_PATTERNS]
        self._cross_ref_patterns = [re.compile(p) 
                                   for p in self.CROSS_REF_PATTERNS]
        self._header_patterns = [re.compile(p, re.MULTILINE | re.IGNORECASE)
                                for p in self.HEADER_FOOTER_PATTERNS]
    
    def build_context(self, text: str, title: str = "", 
                      metadata: Optional[dict] = None) -> dict:
        """Build minimal context for a worker.
        
        Args:
            text: Raw clause text.
            title: Optional clause title.
            metadata: Optional task-specific metadata.
            
        Returns:
            Dict with sanitized text, title, and metadata.
        """
        # Sanitize text
        clean_text = self._sanitize(text)
        
        # Truncate to limit
        clean_text = self._truncate(clean_text, self.config.max_text_length)
        
        # Clean title
        clean_title = title.strip()[:self.config.max_title_length] if title else ""
        
        return {
            "text": clean_text,
            "title": clean_title,
            "metadata": metadata or {},
            "original_length": len(text),
            "sanitized_length": len(clean_text),
            "was_truncated": len(text) > self.config.max_text_length,
        }
    
    def _sanitize(self, text: str) -> str:
        """Remove noise from text."""
        result = text
        
        # Strip page numbers
        if self.config.strip_page_numbers:
            for pattern in self._page_patterns:
                result = pattern.sub('', result)
        
        # Strip cross-references (cause hallucination)
        if self.config.strip_cross_references:
            for pattern in self._cross_ref_patterns:
                result = pattern.sub('[REF]', result)
        
        # Strip headers/footers
        if self.config.strip_headers_footers:
            for pattern in self._header_patterns:
                result = pattern.sub('', result)
        
        # Normalize whitespace
        if self.config.normalize_whitespace:
            result = re.sub(r'\n{3,}', '\n\n', result)
            result = re.sub(r'[ \t]+', ' ', result)
            result = result.strip()
        
        return result
    
    def _truncate(self, text: str, max_length: int) -> str:
        """Truncate text to max length at word boundary."""
        if len(text) <= max_length:
            return text
        
        # Find last space before limit
        truncated = text[:max_length]
        last_space = truncated.rfind(' ')
        
        if last_space > max_length * 0.8:  # Only if reasonable
            truncated = truncated[:last_space]
        
        return truncated + "..."


# Singleton instance
context_builder = ContextBuilder()
