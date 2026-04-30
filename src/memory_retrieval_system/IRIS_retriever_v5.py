#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IRR Retriever V5 - deeply optimized version

Main improvements (vs v3):
1. Entity tracking - resolves entity confusion (Cat5)
2. "Not mentioned" detection - avoids over-speculation (Cat5)
3. Enhanced temporal reasoning - INFERRABLE tier, allows reasonable temporal inference (Cat2)
4. Multi-hop reasoning chain - supports complex questions (Cat5)
5. Dual-path retrieval - improves recall (all categories)
6. Answer certainty optimization - reduces over-cautious phrasing (Cat4)

Carried over from v3:
1. Temporal question detection and dedicated handling
2. Two-tier sufficiency check
3. Approximate reasoning mode
4. Dynamic threshold adjustment
"""

import re
import time
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field

from .retriever import MemoryRetriever


@dataclass
class EntityInfo:
    """Entity information tracking."""
    name: str
    facts: List[str] = field(default_factory=list)
    attributes: Set[str] = field(default_factory=set)


@dataclass
class ReasoningChain:
    """Reasoning chain."""
    question: str
    steps: List[str] = field(default_factory=list)
    entities_involved: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class RetrievalIteration:
    """Result of a single retrieval iteration."""
    iteration: int
    query_original: str  # original question query
    query_refined: str   # refined query
    facts_from_original: List[str]  # facts from original query
    facts_from_refined: List[str]   # facts from refined query
    facts_expanded: List[str]
    has_exact_match: bool
    has_partial_match: bool
    has_inferrable_info: bool  # new: whether information is inferrable
    confidence: float
    missing_info: str = ""
    sufficiency_prompt: str = ""
    sufficiency_response: str = ""
    refine_prompt: str = ""
    refine_response: str = ""


@dataclass
class IRRResult:
    """Complete IRR result."""
    question: str
    final_answer: str
    confidence: float
    is_approximate: bool
    is_not_mentioned: bool  # new: whether the answer is "Not mentioned"
    is_temporal_question: bool
    entities_tracked: List[str]  # new: tracked entities
    reasoning_chain: Optional[ReasoningChain]  # new: reasoning chain
    iterations: List[RetrievalIteration]
    total_facts: int
    retrieval_time: float
    generation_time: float
    total_time: float


class IRRRetrieverV5:
    """
    IRR Retriever V5 - deeply optimized version.

    Core capabilities:
    1. Entity-level information tracking
    2. Smart "Not mentioned" detection
    3. Enhanced temporal reasoning (EXACT/INFERRABLE/INSUFFICIENT)
    4. Multi-hop reasoning chain construction
    5. Dual-path retrieval + merging
    6. Adaptive answer certainty
    """

    # Temporal keywords (inherited from v3)
    TEMPORAL_KEYWORDS = [
        r'\bwhen\b', r'\bwhat\s+time\b', r'\bwhat\s+date\b',
        r'\bwhich\s+date\b', r'\bwhich\s+year\b', r'\bwhat\s+year\b',
        r'\bhow\s+long\b', r'\bhow\s+many\s+(days|months|years)\b',
        r'\bstart\s+(date|time)\b', r'\bend\s+(date|time)\b',
        r'\bdid.*when\b', r'\bwas.*when\b', r'\bwill.*when\b'
    ]

    # Temporal relation words
    TEMPORAL_RELATIONS = {
        'as of': 'current_time',
        'after': 'after',
        'before': 'before',
        'during': 'during',
        'while': 'simultaneous',
        'since': 'since',
        'until': 'until',
        'by': 'deadline'
    }

    # Common entity name patterns (extensible)
    COMMON_NAMES = [
        r'\bJon\b', r'\bGina\b', r'\bJean\b', r'\bJohn\b',
        r'\bMary\b', r'\bDavid\b', r'\bSarah\b', r'\bMike\b'
    ]

    def __init__(
        self,
        base_retriever: MemoryRetriever,
        llm_client,
        provider: str = "openai",
        max_iterations: int = 3,
        sufficiency_threshold: float = 0.7,
        temporal_threshold: float = 0.85,
        inferrable_threshold: float = 0.7,  # new: threshold for inferrable info
        not_mentioned_threshold: float = 0.3,  # new: "Not mentioned" detection threshold
        allow_approximate_reasoning: bool = True,
        temporal_allow_inference: bool = True,  # new: allow temporal inference
        enable_entity_tracking: bool = True,  # new: enable entity tracking
        enable_dual_retrieval: bool = True,  # new: enable dual-path retrieval
        retrieval_mode: str = "index",  # ablation: 'index' or 'raw'
        approximate_confidence_penalty: float = 0.15,  # reduced penalty
        # calibration sensitivity parameters
        calib_exact_floor: float = 0.85,
        calib_inferrable_cap: float = 0.75,
        calib_partial_cap: float = 0.50,
        verbose: bool = True
    ):
        """
        Args:
            base_retriever: underlying MemoryRetriever instance
            llm_client: LLM client
            provider: 'openai' or 'google_ai'
            max_iterations: maximum number of iterations
            sufficiency_threshold: sufficiency threshold for general questions
            temporal_threshold: exact-match threshold for temporal questions
            inferrable_threshold: threshold for inferrable info
            not_mentioned_threshold: "Not mentioned" detection threshold
            allow_approximate_reasoning: whether to allow approximate reasoning
            temporal_allow_inference: whether temporal questions allow inference
            enable_entity_tracking: whether to enable entity tracking
            enable_dual_retrieval: whether to enable dual-path retrieval
            approximate_confidence_penalty: confidence penalty for approximate answers
            verbose: whether to print logs
        """
        self.retriever = base_retriever
        self.llm = llm_client
        self.provider = provider
        self.max_iterations = max_iterations
        self.sufficiency_threshold = sufficiency_threshold
        self.temporal_threshold = temporal_threshold
        self.inferrable_threshold = inferrable_threshold
        self.not_mentioned_threshold = not_mentioned_threshold
        self.allow_approximate = allow_approximate_reasoning
        self.temporal_allow_inference = temporal_allow_inference
        self.enable_entity_tracking = enable_entity_tracking
        self.enable_dual_retrieval = enable_dual_retrieval
        self.retrieval_mode = retrieval_mode
        self.approx_penalty = approximate_confidence_penalty
        self.calib_exact_floor = calib_exact_floor
        self.calib_inferrable_cap = calib_inferrable_cap
        self.calib_partial_cap = calib_partial_cap
        self.verbose = verbose

        # Entity tracking dict
        self.entity_tracker: Dict[str, EntityInfo] = {}

    def _call_llm(self, system_prompt: str, user_prompt: str, model: str,
                  max_tokens: int = 512, temperature: float = 0.1) -> str:
        """Unified LLM invocation interface."""
        if self.provider == "openai":
            response = self.llm.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()

        elif self.provider == "google_ai":
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            response = self.llm.generate_content(
                full_prompt,
                generation_config={
                    "max_output_tokens": max_tokens,
                    "temperature": temperature,
                }
            )
            return response.text.strip()

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _log(self, msg: str):
        if self.verbose:
            try:
                print(f"[IRRv5] {msg}")
            except UnicodeEncodeError:
                safe_msg = msg.encode('ascii', 'replace').decode('ascii')
                print(f"[IRRv5] {safe_msg}")

    def _is_temporal_question(self, question: str) -> bool:
        """Detect whether the question is temporal."""
        question_lower = question.lower()
        for pattern in self.TEMPORAL_KEYWORDS:
            if re.search(pattern, question_lower):
                return True
        return False

    def _extract_entities(self, question: str, model: str = "gpt-4o-mini") -> List[str]:
        """
        Extract entities (people, organizations, etc.) from the question.

        Returns:
            List of entities
        """
        if not self.enable_entity_tracking:
            return []

        # Simple regex match
        entities = []
        for pattern in self.COMMON_NAMES:
            matches = re.findall(pattern, question, re.IGNORECASE)
            entities.extend([m.capitalize() for m in matches])

        # Use LLM for further extraction
        if not entities:
            prompt = f"""Extract all entity names (people, organizations, places) from this question.
Return only the names, comma-separated, or "none" if no entities.

Question: {question}

Entities:"""
            try:
                result = self._call_llm(
                    system_prompt="You are a helpful assistant that extracts entities.",
                    user_prompt=prompt,
                    model=model,
                    max_tokens=50,
                    temperature=0
                )
                if result.lower() != "none":
                    entities = [e.strip() for e in result.split(",")]
            except:
                pass

        return list(set(entities))

    def _track_entity_facts(self, entity: str, facts: List[str]):
        """Track facts associated with a specific entity."""
        if entity not in self.entity_tracker:
            self.entity_tracker[entity] = EntityInfo(name=entity)

        entity_facts = []
        for fact in facts:
            if entity.lower() in fact.lower():
                entity_facts.append(fact)
                self.entity_tracker[entity].facts.append(fact)

        return entity_facts

    def _detect_not_mentioned(
        self,
        question: str,
        all_facts: List[str],
        has_exact: bool,
        has_partial: bool,
        has_inferrable: bool,
        confidence: float,
        iterations_done: int
    ) -> bool:
        """
        Decide whether the answer should be "Not mentioned".

        Conditions:
        1. No matches at all (no exact/partial/inferrable)
        2. OR: very low confidence (<0.3) with few retrieved facts
        3. AND: already tried multiple iterations

        Returns:
            True if should answer "Not mentioned"
        """
        # Condition 1: no matches whatsoever
        if not has_exact and not has_partial and not has_inferrable:
            if iterations_done >= 2:  # at least 2 iterations
                self._log("[Not Mentioned] No matches found after iterations")
                return True

        # Condition 2: very low confidence + few facts
        if confidence < self.not_mentioned_threshold and len(all_facts) < 3:
            if iterations_done >= self.max_iterations:
                self._log(f"[Not Mentioned] Low confidence ({confidence:.2f}) with few facts")
                return True

        return False

    def _check_sufficiency_v5(
        self,
        question: str,
        facts: List[str],
        is_temporal: bool,
        model: str = "gpt-4o-mini"
    ) -> Tuple[bool, bool, bool, float, str, str, str]:
        """
        V5 sufficiency check: distinguishes EXACT/INFERRABLE/INSUFFICIENT.

        Returns:
            (has_exact, has_partial, has_inferrable, confidence, missing_info, prompt_text, response_text)
        """
        if not facts:
            return False, False, False, 0.0, "No facts retrieved", "", ""

        facts_text = "\n".join(f"- {f}" for f in facts[:20])

        # Special instruction for temporal questions
        extra_instruction = ""
        if is_temporal:
            extra_instruction = """
IMPORTANT: This is a TEMPORAL question asking for specific dates/times.
- EXACT_MATCH: Has precise date/time (e.g., "January 19, 2023", "2023-01-19")
- INFERRABLE: Has temporal clues that allow reasonable inference
  * Example: "as of February 2023" → event likely in February 2023
  * Example: "after opening in January" → subsequent events after January
- PARTIAL_MATCH: Has related but insufficient temporal information
- Vague terms like "recently", "not long ago" are PARTIAL, not EXACT
"""

        prompt = f"""Question: {question}

Retrieved Facts:
{facts_text}
{extra_instruction}

Evaluate if these facts can answer the question:

1. EXACT_MATCH: Can answer precisely with specific details? (yes/no)
2. INFERRABLE: Can reasonably infer the answer from related information? (yes/no)
3. PARTIAL_MATCH: Have related information but not enough to answer? (yes/no)
4. CONFIDENCE: Your confidence level (0.0-1.0)
5. MISSING: What specific information is missing? (or "none" if exact/inferrable)

Respond in EXACTLY this format:
EXACT: yes/no
INFERRABLE: yes/no
PARTIAL: yes/no
CONFIDENCE: 0.0-1.0
MISSING: <missing information or "none">"""

        try:
            result = self._call_llm(
                system_prompt="You are a helpful assistant that evaluates information sufficiency.",
                user_prompt=prompt,
                model=model,
                max_tokens=300,
                temperature=0
            )

            # Parse result
            exact = "yes" in result.lower().split("exact:")[1].split("\n")[0] if "exact:" in result.lower() else False
            inferrable = "yes" in result.lower().split("inferrable:")[1].split("\n")[0] if "inferrable:" in result.lower() else False
            partial = "yes" in result.lower().split("partial:")[1].split("\n")[0] if "partial:" in result.lower() else False

            confidence = 0.5
            if "confidence:" in result.lower():
                try:
                    conf_str = result.lower().split("confidence:")[1].split("\n")[0].strip()
                    confidence = float(conf_str)
                except:
                    pass

            # Adjust confidence
            if is_temporal:
                if exact:
                    confidence = max(confidence, self.calib_exact_floor)
                elif inferrable and self.temporal_allow_inference:
                    confidence = min(confidence, self.calib_inferrable_cap)
                    self._log(f"[Temporal-Inferrable] Confidence capped at {confidence:.2f}")
                elif partial:
                    confidence = min(confidence, self.calib_partial_cap)
                    self._log(f"[Temporal-Partial] Confidence lowered to {confidence:.2f}")

            missing = ""
            if "missing:" in result.lower():
                missing = result.lower().split("missing:")[1].strip().split("\n")[0]
                if missing == "none":
                    missing = ""

            return exact, partial, inferrable, confidence, missing, prompt, result

        except Exception as e:
            self._log(f"Sufficiency check failed: {e}")
            return False, False, False, 0.5, "evaluation failed", prompt if 'prompt' in dir() else "", ""

    def _build_reasoning_chain(
        self,
        question: str,
        facts: List[str],
        entities: List[str],
        model: str = "gpt-4o-mini"
    ) -> Optional[ReasoningChain]:
        """
        Build reasoning chain (for multi-hop questions).

        Example:
        Q: "What dance piece did Jon's team perform to win first place?"
        Chain:
        1. Identify Jon's team
        2. Find competition where they won first place
        3. Find the dance piece performed in that competition
        """
        if len(facts) < 2:
            return None

        facts_text = "\n".join(f"- {f}" for f in facts[:15])
        entities_text = ", ".join(entities) if entities else "none"

        prompt = f"""Analyze if this question requires multi-hop reasoning.

Question: {question}
Entities: {entities_text}

Facts available:
{facts_text}

If this requires connecting multiple facts (multi-hop reasoning):
1. List the reasoning steps needed
2. Identify which facts support each step

If it's a simple direct question, respond "DIRECT".

Response format:
TYPE: MULTI-HOP / DIRECT
STEPS:
- Step 1: [description]
- Step 2: [description]
...
FACTS_USED: [comma-separated fact indices]"""

        try:
            result = self._call_llm(
                system_prompt="You are a helpful assistant that analyzes reasoning requirements.",
                user_prompt=prompt,
                model=model,
                max_tokens=400,
                temperature=0
            )

            if "TYPE: DIRECT" in result or "DIRECT" in result.upper():
                return None

            # Parse reasoning steps
            steps = []
            if "STEPS:" in result:
                steps_section = result.split("STEPS:")[1].split("FACTS_USED:")[0] if "FACTS_USED:" in result else result.split("STEPS:")[1]
                for line in steps_section.strip().split("\n"):
                    if line.strip().startswith("-"):
                        steps.append(line.strip()[1:].strip())

            if steps:
                return ReasoningChain(
                    question=question,
                    steps=steps,
                    entities_involved=entities,
                    confidence=0.7
                )

            return None

        except Exception as e:
            self._log(f"Reasoning chain building failed: {e}")
            return None

    def _retrieve_dual_path(
        self,
        original_query: str,
        refined_query: str,
        top_k: int,
        expand_hops: int
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Dual-path retrieval: use both the original question and the refined query.

        Returns:
            (facts_from_original, facts_from_refined, expanded_facts)
        """
        # ===== Ablation mode: Raw BM25 =====
        if self.retrieval_mode == "raw":
            self._log(f"[Raw BM25 Retrieval] query={refined_query[:80]}...")
            if not self.enable_dual_retrieval or original_query == refined_query:
                results = self.retriever.search_raw_bm25(refined_query, limit=top_k)
                facts_refined = [r.raw_text for r, _ in results]
                return [], facts_refined, []
            # dual-path on raw
            results_orig = self.retriever.search_raw_bm25(original_query, limit=max(5, top_k // 2))
            results_ref = self.retriever.search_raw_bm25(refined_query, limit=top_k)
            facts_original = [r.raw_text for r, _ in results_orig]
            facts_refined = [r.raw_text for r, _ in results_ref]
            self._log(f"[Raw Dual Retrieval] Original: {len(facts_original)}, Refined: {len(facts_refined)}")
            return facts_original, facts_refined, []

        # ===== Normal mode: Index + Edge =====
        if not self.enable_dual_retrieval or original_query == refined_query:
            # Single-path retrieval
            result = self.retriever.retrieve(
                query=refined_query,
                top_k=top_k,
                hops=expand_hops,
                include_raw=False,
                search_method="auto"
            )
            facts_refined = [idx.to_fact_string() for idx in result.indexes]
            expanded = [idx.to_fact_string() for idx in result.expanded_indexes]
            return [], facts_refined, expanded

        # Dual-path retrieval
        result_orig = self.retriever.retrieve(
            query=original_query,
            top_k=max(5, top_k // 2),
            hops=expand_hops,
            include_raw=False,
            search_method="auto"
        )
        result_refined = self.retriever.retrieve(
            query=refined_query,
            top_k=top_k,
            hops=expand_hops,
            include_raw=False,
            search_method="auto"
        )

        facts_original = [idx.to_fact_string() for idx in result_orig.indexes]
        facts_refined = [idx.to_fact_string() for idx in result_refined.indexes]

        # Merge expanded facts
        expanded_orig = [idx.to_fact_string() for idx in result_orig.expanded_indexes]
        expanded_refined = [idx.to_fact_string() for idx in result_refined.expanded_indexes]
        expanded = list(dict.fromkeys(expanded_orig + expanded_refined))

        self._log(f"[Dual Retrieval] Original: {len(facts_original)}, Refined: {len(facts_refined)}, Expanded: {len(expanded)}")

        return facts_original, facts_refined, expanded

    def _refine_query_adaptive(
        self,
        original_question: str,
        current_query: str,
        missing_info: str,
        is_temporal: bool,
        iteration: int,
        model: str = "gpt-4o-mini"
    ) -> Tuple[str, str, str]:
        """
        Adaptive query refinement: adjust strategy by question type and iteration.

        Returns:
            (refined_query, prompt_text, response_text)
        """
        if not missing_info or missing_info == "none":
            return current_query, "", ""

        if is_temporal:
            # Temporal-question strategy
            if iteration == 1:
                strategy = "Focus on DATE, TIME, and temporal keywords (when, started, launched, opened)."
            elif iteration == 2:
                strategy = "Search for specific DATE FORMATS and temporal relations (as of, after, before, during)."
            else:
                strategy = "Try broader temporal context: related events, milestones, timeframes."
        else:
            # Non-temporal strategy
            if iteration == 1:
                strategy = "Focus on specific keywords, entity names, and key concepts."
            elif iteration == 2:
                strategy = "Try different angle: related events, attributes, or contextual information."
            else:
                strategy = "Use synonyms, broader concepts, or implied relationships."

        prompt = f"""Original question: {original_question}
Current search query: {current_query}
Missing information: {missing_info}
Iteration: {iteration}/{self.max_iterations}

Strategy: {strategy}

Generate an improved search query to find the missing information.
Keep it concise and focused. Return ONLY the query (no explanation)."""

        try:
            refined = self._call_llm(
                system_prompt="You are a helpful assistant that refines search queries.",
                user_prompt=prompt,
                model=model,
                max_tokens=100,
                temperature=0.2
            )
            refined = refined.strip('"\'')
            return (refined if refined else current_query), prompt, refined
        except Exception as e:
            self._log(f"Query refinement failed: {e}")
            return current_query, prompt if 'prompt' in dir() else "", ""

    def _generate_answer_v5(
        self,
        question: str,
        all_facts: List[str],
        has_exact: bool,
        has_partial: bool,
        has_inferrable: bool,
        is_temporal: bool,
        is_not_mentioned: bool,
        reasoning_chain: Optional[ReasoningChain],
        confidence: float,
        model: str = "gpt-4o"
    ) -> Tuple[str, float, bool]:
        """
        V5 answer generation: adaptive certainty, supports reasoning chains.

        Returns:
            (answer, confidence, is_approximate)
        """
        # "Not mentioned" case
        if is_not_mentioned:
            return "Not mentioned", 0.1, False

        if not all_facts:
            return "I don't have enough information to answer this question.", 0.0, False

        unique_facts = list(dict.fromkeys(all_facts))[:30]
        facts_text = "\n".join(f"- {f}" for f in unique_facts)

        # Reasoning-chain context
        reasoning_context = ""
        if reasoning_chain and reasoning_chain.steps:
            steps_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(reasoning_chain.steps))
            reasoning_context = f"\n\nReasoning steps required:\n{steps_text}\n"

        # Adjust answer strategy based on sufficiency and confidence
        is_approximate = False

        if is_temporal:
            # Temporal question
            if has_exact:
                instruction = """This is a TEMPORAL question. Answer with the PRECISE date/time from the facts.
Use standard formats: "January 19, 2023", "2023-01-19", "May 2023".
Do NOT use vague terms like "around", "approximately"."""
                is_approximate = False

            elif has_inferrable and self.temporal_allow_inference:
                instruction = """The facts contain temporal clues that allow REASONABLE INFERENCE.
Make a careful inference:
- If "as of February 2023", the event likely happened in/around February 2023
- If "after opening in January", the event is after January
- Be clear but concise. Use "in" or "around" if inferring timeframe
- Do NOT be overly cautious with phrases like "it is likely that" or "based on the facts"."""
                is_approximate = True

            else:
                instruction = "The facts don't contain sufficient temporal information. State what's missing briefly."
                is_approximate = False

        else:
            # Non-temporal: adjust certainty by confidence
            if has_exact or (has_inferrable and confidence >= 0.75):
                # High confidence: answer directly
                instruction = """Answer the question directly and confidently using the facts.
Do NOT use hedging phrases like "likely", "it seems", "probably", "based on the facts".
If you have the information, state it clearly."""
                is_approximate = False

            elif has_inferrable or (has_partial and confidence >= 0.5):
                # Medium confidence: inference-based
                instruction = """Answer based on reasonable inference from the facts.
Use clear language. If inferring, use:
- "Based on the facts, [answer]" OR
- Simply state the answer with brief reasoning
Do NOT overuse uncertainty markers ("likely", "it seems", "probably")."""
                is_approximate = True

            else:
                # Low confidence: explain missing info
                instruction = "Answer based on available facts. If key information is missing, state it concisely."
                is_approximate = False

        prompt = f"""{instruction}
{reasoning_context}
Question: {question}

Relevant Facts:
{facts_text}

Answer (be concise and direct):"""

        try:
            answer = self._call_llm(
                system_prompt="You are a helpful assistant that answers questions based on provided facts.",
                user_prompt=prompt,
                model=model,
                max_tokens=350,
                temperature=0
            )

            # Compute final confidence
            if is_approximate:
                final_confidence = max(0.3, confidence - self.approx_penalty)
            else:
                final_confidence = confidence

            return answer, final_confidence, is_approximate

        except Exception as e:
            self._log(f"Answer generation failed: {e}")
            return f"Error generating answer: {e}", 0.0, False

    def retrieve_with_irr(
        self,
        question: str,
        base_top_k: int = 10,
        expand_hops: int = 1,
        model: str = "gpt-4o"
    ) -> IRRResult:
        """
        Execute the IRRv5 retrieval pipeline.

        Full pipeline:
        1. Entity extraction and tracking
        2. Temporal-question detection
        3. Dual-path iterative retrieval
        4. Sufficiency check (EXACT/INFERRABLE/PARTIAL)
        5. "Not mentioned" detection
        6. Multi-hop reasoning chain construction
        7. Adaptive answer generation

        Returns:
            IRRResult with the full result
        """
        # 1. Entity extraction
        entities = self._extract_entities(question)
        if entities:
            self._log(f"Entities extracted: {entities}")

        # 2. Temporal-question detection
        is_temporal = self._is_temporal_question(question)
        temporal_tag = "[TEMPORAL]" if is_temporal else "[GENERAL]"

        self._log(f"\n{'='*60}")
        self._log(f"IRRv5 Started {temporal_tag}: {question}")
        self._log(f"{'='*60}")

        start_time = time.time()
        iterations: List[RetrievalIteration] = []
        all_facts_from_original: List[str] = []
        all_facts_from_refined: List[str] = []
        all_expanded: List[str] = []

        # Initial query
        current_query = question
        has_exact = False
        has_partial = False
        has_inferrable = False
        confidence = 0.0
        missing_info = ""

        # 3. Iterative retrieval
        retrieval_start = time.time()

        for i in range(1, self.max_iterations + 1):
            self._log(f"\n--- Iteration {i}/{self.max_iterations} ---")

            # Dual-path retrieval
            facts_orig, facts_ref, expanded = self._retrieve_dual_path(
                original_query=question,
                refined_query=current_query,
                top_k=base_top_k + (i - 1) * 3,
                expand_hops=expand_hops
            )

            all_facts_from_original.extend(facts_orig)
            all_facts_from_refined.extend(facts_ref)
            all_expanded.extend(expanded)

            # Merge all facts
            current_iteration_facts = list(dict.fromkeys(facts_orig + facts_ref + expanded))
            all_facts_combined = list(dict.fromkeys(
                all_facts_from_original + all_facts_from_refined + all_expanded
            ))

            self._log(f"Facts retrieved: {len(current_iteration_facts)} (Total: {len(all_facts_combined)})")

            # Entity tracking
            if entities and self.enable_entity_tracking:
                for entity in entities:
                    entity_facts = self._track_entity_facts(entity, current_iteration_facts)
                    if entity_facts:
                        self._log(f"Entity '{entity}' facts: {len(entity_facts)}")

            # Sufficiency check
            has_exact, has_partial, has_inferrable, confidence, missing_info, suf_prompt, suf_response = self._check_sufficiency_v5(
                question=question,
                facts=all_facts_combined,
                is_temporal=is_temporal
            )

            self._log(f"Sufficiency: Exact={has_exact}, Inferrable={has_inferrable}, Partial={has_partial}")
            self._log(f"Confidence: {confidence:.2f}")
            if missing_info:
                self._log(f"Missing: {missing_info}")

            # Save iteration result
            iterations.append(RetrievalIteration(
                iteration=i,
                query_original=question,
                query_refined=current_query,
                facts_from_original=facts_orig,
                facts_from_refined=facts_ref,
                facts_expanded=expanded,
                has_exact_match=has_exact,
                has_partial_match=has_partial,
                has_inferrable_info=has_inferrable,
                confidence=confidence,
                missing_info=missing_info,
                sufficiency_prompt=suf_prompt,
                sufficiency_response=suf_response
            ))

            # Decide threshold
            current_threshold = self.temporal_threshold if is_temporal else self.sufficiency_threshold

            # Termination conditions
            terminate = False
            if has_exact and confidence >= current_threshold:
                self._log(f"[Terminate] Exact match with sufficient confidence ({confidence:.2f} >= {current_threshold})")
                terminate = True
            elif has_inferrable and confidence >= self.inferrable_threshold and self.temporal_allow_inference:
                self._log(f"[Terminate] Inferrable with sufficient confidence ({confidence:.2f} >= {self.inferrable_threshold})")
                terminate = True
            elif i >= self.max_iterations:
                self._log(f"[Terminate] Max iterations reached")
                terminate = True

            if terminate:
                break

            # Refine query for next iteration
            if missing_info and i < self.max_iterations:
                current_query, ref_prompt, ref_response = self._refine_query_adaptive(
                    original_question=question,
                    current_query=current_query,
                    missing_info=missing_info,
                    is_temporal=is_temporal,
                    iteration=i
                )
                # Save refine prompt/response on current iteration
                iterations[-1].refine_prompt = ref_prompt
                iterations[-1].refine_response = ref_response
                self._log(f"Refined query: {current_query}")

        retrieval_time = time.time() - retrieval_start

        # 4. "Not mentioned" detection
        all_facts_final = list(dict.fromkeys(
            all_facts_from_original + all_facts_from_refined + all_expanded
        ))

        is_not_mentioned = self._detect_not_mentioned(
            question=question,
            all_facts=all_facts_final,
            has_exact=has_exact,
            has_partial=has_partial,
            has_inferrable=has_inferrable,
            confidence=confidence,
            iterations_done=len(iterations)
        )

        # 5. Build reasoning chain (if needed)
        reasoning_chain = None
        if not is_not_mentioned and (has_inferrable or has_partial):
            reasoning_chain = self._build_reasoning_chain(
                question=question,
                facts=all_facts_final[:20],
                entities=entities
            )
            if reasoning_chain:
                self._log(f"Reasoning chain built: {len(reasoning_chain.steps)} steps")

        # 6. Generate answer
        gen_start = time.time()
        final_answer, final_confidence, is_approximate = self._generate_answer_v5(
            question=question,
            all_facts=all_facts_final,
            has_exact=has_exact,
            has_partial=has_partial,
            has_inferrable=has_inferrable,
            is_temporal=is_temporal,
            is_not_mentioned=is_not_mentioned,
            reasoning_chain=reasoning_chain,
            confidence=confidence,
            model=model
        )
        generation_time = time.time() - gen_start

        total_time = time.time() - start_time

        self._log(f"\n{'='*60}")
        self._log(f"IRRv5 Completed in {total_time:.2f}s")
        self._log(f"Answer: {final_answer[:100]}...")
        self._log(f"Confidence: {final_confidence:.2f}")
        self._log(f"Approximate: {is_approximate}, Not Mentioned: {is_not_mentioned}")
        self._log(f"{'='*60}\n")

        return IRRResult(
            question=question,
            final_answer=final_answer,
            confidence=final_confidence,
            is_approximate=is_approximate,
            is_not_mentioned=is_not_mentioned,
            is_temporal_question=is_temporal,
            entities_tracked=entities,
            reasoning_chain=reasoning_chain,
            iterations=iterations,
            total_facts=len(all_facts_final),
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            total_time=total_time
        )


def create_irr_retriever_v5(
    db_path: str,
    openai_api_key: str,
    provider: str = "openai",
    model: str = "gpt-4o",
    max_iterations: int = 3,
    allow_approximate_reasoning: bool = True,
    temporal_allow_inference: bool = True,
    enable_entity_tracking: bool = True,
    enable_dual_retrieval: bool = True,
    retrieval_mode: str = "index",
    calib_exact_floor: float = 0.85,
    calib_inferrable_cap: float = 0.75,
    calib_partial_cap: float = 0.50,
    verbose: bool = True
) -> IRRRetrieverV5:
    """
    Factory function for creating an IRRv5 instance.

    Args:
        db_path: database path
        openai_api_key: OpenAI API key
        provider: 'openai' or 'google_ai'
        model: model name
        max_iterations: maximum number of iterations
        allow_approximate_reasoning: whether to allow approximate reasoning
        temporal_allow_inference: whether temporal questions allow inference
        enable_entity_tracking: whether to enable entity tracking
        enable_dual_retrieval: whether to enable dual-path retrieval
        verbose: whether to print verbose logs

    Returns:
        IRRRetrieverV5 instance
    """
    from .retriever import MemoryRetriever

    # Ensure db_path is in SQLAlchemy URL format
    if not db_path.startswith("sqlite:///"):
        db_path = f"sqlite:///{db_path}"

    # Ensure API key is set as an env var
    import os
    if provider == "openai":
        os.environ["OPENAI_API_KEY"] = openai_api_key
    elif provider == "google_ai":
        os.environ["GEMINI_API_KEY"] = openai_api_key  # the param is named openai_api_key but it's reused for gemini

    # Create underlying retriever
    base_retriever = MemoryRetriever(db_path, echo=False, embedding_provider=provider)

    # Create LLM client
    if provider == "openai":
        from openai import OpenAI
        llm_client = OpenAI(api_key=openai_api_key)
    elif provider == "google_ai":
        import google.generativeai as genai
        genai.configure(api_key=openai_api_key)
        llm_client = genai.GenerativeModel(model)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    # Create IRRv5 instance
    return IRRRetrieverV5(
        base_retriever=base_retriever,
        llm_client=llm_client,
        provider=provider,
        max_iterations=max_iterations,
        allow_approximate_reasoning=allow_approximate_reasoning,
        temporal_allow_inference=temporal_allow_inference,
        enable_entity_tracking=enable_entity_tracking,
        enable_dual_retrieval=enable_dual_retrieval,
        retrieval_mode=retrieval_mode,
        calib_exact_floor=calib_exact_floor,
        calib_inferrable_cap=calib_inferrable_cap,
        calib_partial_cap=calib_partial_cap,
        verbose=verbose
    )
