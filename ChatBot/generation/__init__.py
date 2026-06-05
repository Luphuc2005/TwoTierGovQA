# Generation Module for Legal RAG System
# Bao gồm: Context Builder, Prompt Templates, Gating, LLM Client, Evaluator, Fallback

from .rag_contract import (
    RAGInput, RAGOutput, ChunkInfo, Citation, 
    RAGPolicy, DecisionType, AbstainReason, LLMTier,
    create_rag_input, parse_rag_output, compute_citation_correctness
)
from .context_builder import ContextBuilder, ContextOptimizer, build_context_for_generation
from .prompt_templates import LegalPromptBuilder, PromptConfig, PromptStyle, SpecializedPrompts
from .gating import GatingStrategy, GatingConfig, GatingDecision, ScoreCalibrator
from .llm_client import LLMClient, LLMConfig, LLMBackend, LLMMode, LLMResponse, select_local_model_for_vram
from .fallback import FallbackStrategy, FallbackDecision, FallbackType, apply_fallback_if_needed
from .pipeline_logger import PipelineLogger, PipelineLogEntry
from .evaluator import (
    GenerationEvaluator, EvalSample, EvalResult, EvalSummary,
    load_goldset, create_goldset_from_eval_qa
)

__all__ = [
    # Contract
    "RAGInput", "RAGOutput", "ChunkInfo", "Citation", "RAGPolicy",
    "DecisionType", "AbstainReason", "LLMTier",
    "create_rag_input", "parse_rag_output", "compute_citation_correctness",
    
    # Context
    "ContextBuilder", "ContextOptimizer", "build_context_for_generation",
    
    # Prompt
    "LegalPromptBuilder", "PromptConfig", "PromptStyle", "SpecializedPrompts",
    
    # Gating
    "GatingStrategy", "GatingConfig", "GatingDecision", "ScoreCalibrator",
    
    # LLM
    "LLMClient", "LLMConfig", "LLMBackend", "LLMMode", "LLMResponse",
    
    # Fallback
    "FallbackStrategy", "FallbackDecision", "FallbackType", "apply_fallback_if_needed",
    
    # Logger
    "PipelineLogger", "PipelineLogEntry",
    
    # Evaluation
    "GenerationEvaluator", "EvalSample", "EvalResult", "EvalSummary",
    "load_goldset", "create_goldset_from_eval_qa"
]
