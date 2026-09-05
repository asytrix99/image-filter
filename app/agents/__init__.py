"""Module 6 - Strands multi-agent layer.

An orchestrator routes natural-language requests to specialist agents
(search / curator / ingestion), each of which wraps the Phase 1-2 pipeline
functions as tools. The LLM makes decisions; the heavy CV work stays in the
plain Python tools.
"""
