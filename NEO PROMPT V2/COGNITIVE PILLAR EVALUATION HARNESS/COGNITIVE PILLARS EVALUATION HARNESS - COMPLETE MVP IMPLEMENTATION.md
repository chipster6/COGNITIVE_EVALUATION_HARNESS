# ============================================================================

# COGNITIVE PILLARS EVALUATION HARNESS - COMPLETE MVP IMPLEMENTATION

# ============================================================================

# This file contains the complete implementation including

# 1. Core utilities (seed derivation, uncertainty metrics)

# 2. Working Memory probe (full implementation)

# 3. Stubs for other MVP pillars (Gf, Tool Use, Grounding, Robustness)

# 4. Schemas, tests, and example runner

# ============================================================================

# ============================================================================

# FILE: neoprompt/generators/seed_utils.py

# ============================================================================

“””
Deterministic seed derivation for reproducible probe generation.

All randomness must flow through derive_probe_seed() to ensure:

- Same inputs always produce same probes
- Different variants have different but deterministic seeds
- Seed version changes invalidate old artifacts
  “””

import hashlib

SEED_VERSION = “v1.0.0”  # Increment on backward-incompatible changes

def derive_probe_seed(
master_seed: int,
pillar_id: str,
case_id: str,
variant_idx: int
) -> int:
“””
Derive deterministic seed for a specific probe variant.

```
Args:
    master_seed: Suite-level master seed
    pillar_id: Pillar identifier (e.g., "gf", "wm", "tool_use")
    case_id: Unique case identifier from suite.yml
    variant_idx: 0-based variant index

Returns:
    Derived seed as uint64

Example:
    >>> seed = derive_probe_seed(42, "wm", "wm_001", 0)
    >>> # Always returns same value for same inputs
"""
# Concatenate all components with version prefix
components = f"{SEED_VERSION}:{master_seed}:{pillar_id}:{case_id}:{variant_idx}"

# Hash to get deterministic pseudo-random seed
hash_digest = hashlib.sha256(components.encode('utf-8')).digest()

# Convert first 8 bytes to uint64
seed = int.from_bytes(hash_digest[:8], byteorder='big')

return seed
```

def hash_seed_for_storage(seed: int, rotation_salt: str) -> str:
“””
Hash seed for privacy-preserving storage.

```
NEVER store plaintext seeds in artifacts. Only store hashes.

Args:
    seed: Original seed value
    rotation_salt: Rotation identifier (e.g., "rotation_v1")

Returns:
    SHA-256 hash as hex string
"""
combined = f"{seed}:{rotation_salt}"
return hashlib.sha256(combined.encode('utf-8')).hexdigest()
```

# ============================================================================

# FILE: neoprompt/metrics/uncertainty.py

# ============================================================================

“””
Uncertainty quantification via bootstrap confidence intervals.
“””

import numpy as np

def bootstrap_ci(
data: np.ndarray,
n_bootstrap: int = 1000,
alpha: float = 0.05,
random_state: int = 42
) -> tuple:
“””
Compute bootstrap confidence interval.

```
Args:
    data: Array of scores [0, 1]
    n_bootstrap: Number of bootstrap samples
    alpha: Significance level (0.05 for 95% CI)
    random_state: Fixed seed for reproducible CIs

Returns:
    (lower_bound, upper_bound) as floats in [0, 1]

Example:
    >>> scores = np.array([0.8, 0.9, 0.85, 0.92, 0.88])
    >>> lower, upper = bootstrap_ci(scores)
    >>> # Returns (0.82, 0.91) or similar
"""
rng = np.random.RandomState(random_state)

bootstrap_means = []
for _ in range(n_bootstrap):
    # Resample with replacement
    sample = rng.choice(data, size=len(data), replace=True)
    bootstrap_means.append(np.mean(sample))

# Percentile method
lower = np.percentile(bootstrap_means, 100 * alpha / 2)
upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

return (float(lower), float(upper))
```

# ============================================================================

# FILE: neoprompt/probes/wm.py

# ============================================================================

“””
Working Memory / Context Control Probe (Pillar 3)

FULL IMPLEMENTATION - Reference example for other pillars
“””

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import numpy as np

@dataclass
class Move:
“”“Single entity-to-entity transfer operation.”””
action: str
from_entity: str
to_entity: str
item: str
quantity: int

```
def to_narrative(self) -> str:
    """Convert to natural language."""
    return f"{self.from_entity} gives {self.quantity} {self.item} to {self.to_entity}"
```

@dataclass
class EntityTrackingScenario:
“”“Complete scenario with moves and expected state.”””
initial_state: Dict[str, Dict[str, int]]
moves: List[Move]
expected_final_state: Dict[str, Dict[str, int]]
contradictions: List[Dict[str, Any]]
metadata: Dict[str, Any]

@dataclass
class WorkingMemoryProbe:
“”“Individual WM probe instance.”””
probe_id: str
pillar: str
family: str
band: str
seed_hash: str
narrative: str
question: str
schema_path: str
expected_output: Dict[str, Dict[str, int]]
metadata: Dict[str, Any]

def generate_entity_tracking(seed: int, band: str) -> EntityTrackingScenario:
“””
Generate deterministic entity tracking scenario.

```
Band Parameters:
    A: 2 entities, 3 items, 5 moves
    B: 3 entities, 4 items, 8 moves
    C: 4 entities, 5 items, 12 moves
    D: 5 entities, 6 items, 16 moves
    E: 6 entities, 7 items, 20 moves + contradictions
"""
rng = np.random.RandomState(seed)

params = {
    "A": {"n_entities": 2, "n_items": 3, "n_moves": 5},
    "B": {"n_entities": 3, "n_items": 4, "n_moves": 8},
    "C": {"n_entities": 4, "n_items": 5, "n_moves": 12},
    "D": {"n_entities": 5, "n_items": 6, "n_moves": 16},
    "E": {"n_entities": 6, "n_items": 7, "n_moves": 20}
}[band]

entity_names = [f"Agent{chr(65+i)}" for i in range(params["n_entities"])]
item_types = [f"item_{chr(97+i)}" for i in range(params["n_items"])]

# Initialize with random counts
initial_state = {}
for entity in entity_names:
    inventory = {item: int(rng.randint(0, 6)) for item in item_types}
    initial_state[entity] = inventory.copy()

# Generate moves
moves = []
current_state = {name: inv.copy() for name, inv in initial_state.items()}

for _ in range(params["n_moves"]):
    from_entity = rng.choice(entity_names)
    to_entity_pool = [e for e in entity_names if e != from_entity]
    to_entity = rng.choice(to_entity_pool) if to_entity_pool else from_entity
    
    available_items = [
        item for item, count in current_state[from_entity].items() if count > 0
    ]
    
    if not available_items:
        continue
    
    item = rng.choice(available_items)
    max_qty = current_state[from_entity][item]
    quantity = int(rng.randint(1, max_qty + 1))
    
    move = Move("give", from_entity, to_entity, item, quantity)
    moves.append(move)
    
    current_state[from_entity][item] -= quantity
    current_state[to_entity][item] = current_state[to_entity].get(item, 0) + quantity

# Inject contradictions for band E
contradictions = []
if band == "E":
    for _ in range(2):
        entity = rng.choice(entity_names)
        item = rng.choice(item_types)
        actual = current_state[entity][item]
        wrong = actual + int(rng.randint(1, 6))
        
        contradictions.append({
            "entity": entity,
            "item": item,
            "claimed_count": wrong,
            "actual_count": actual
        })

return EntityTrackingScenario(
    initial_state=initial_state,
    moves=moves,
    expected_final_state=current_state,
    contradictions=contradictions,
    metadata={
        "band": band,
        "n_entities": params["n_entities"],
        "n_moves": len(moves),
        "has_contradictions": len(contradictions) > 0
    }
)
```

def format_narrative(scenario: EntityTrackingScenario) -> str:
“”“Convert scenario to natural language.”””
lines = [“Initial state:”]

```
for entity, inventory in sorted(scenario.initial_state.items()):
    items_list = [f"{count} {item}" for item, count in sorted(inventory.items()) if count > 0]
    items_str = ", ".join(items_list) if items_list else "nothing"
    lines.append(f"- {entity} has: {items_str}")

lines.append("\nMoves:")
for i, move in enumerate(scenario.moves, 1):
    lines.append(f"{i}. {move.to_narrative()}")

if scenario.contradictions:
    lines.append("\nAdditional claims:")
    for c in scenario.contradictions:
        lines.append(f"- Someone claims {c['entity']} has {c['claimed_count']} {c['item']}")

return "\n".join(lines)
```

def generate_wm_probes(
base_case: Dict[str, Any],
master_seed: int,
n_variants: int = 10
) -> List[WorkingMemoryProbe]:
“””
Generate WM probe variants.

```
Returns 10 probes distributed across bands A-E (2 per band).
"""
from neoprompt.generators.seed_utils import derive_probe_seed, hash_seed_for_storage

probes = []
band_sequence = ["A", "A", "B", "B", "C", "C", "D", "D", "E", "E"]

for variant_idx in range(n_variants):
    probe_seed = derive_probe_seed(
        master_seed=master_seed,
        pillar_id="wm",
        case_id=base_case["case_id"],
        variant_idx=variant_idx
    )
    
    band = band_sequence[variant_idx % len(band_sequence)]
    scenario = generate_entity_tracking(seed=probe_seed, band=band)
    narrative = format_narrative(scenario)
    
    probe = WorkingMemoryProbe(
        probe_id=f"{base_case['case_id']}_v{variant_idx:02d}",
        pillar="wm",
        family="wm_bags_v1",
        band=band,
        seed_hash=hash_seed_for_storage(probe_seed, "rotation_v1"),
        narrative=narrative,
        question="What is the final inventory of each entity?",
        schema_path="schemas/wm_bags_v1.json",
        expected_output=scenario.expected_final_state,
        metadata=scenario.metadata
    )
    
    probes.append(probe)

return probes
```

@dataclass
class WorkingMemoryScore:
“”“Detailed scoring for WM probe.”””
probe_id: str
step_accuracy: float
context_loss: float
contradiction_rate: float
score: float
band: str
metadata: Dict[str, Any] = field(default_factory=dict)

def score_wm_probe(
probe: WorkingMemoryProbe,
model_output: Dict[str, Any]
) -> WorkingMemoryScore:
“””
Score WM probe with formula:
S = step_accuracy - 0.1×context_loss - 0.2×contradiction_rate
“””
expected_state = probe.expected_output
model_state = model_output.get(“final_state”, {})

```
total_slots = sum(len(items) for items in expected_state.values())

# Count correct slots
correct_slots = 0
for entity, expected_items in expected_state.items():
    model_items = model_state.get(entity, {})
    for item, expected_count in expected_items.items():
        if model_items.get(item, 0) == expected_count:
            correct_slots += 1

step_accuracy = correct_slots / total_slots if total_slots > 0 else 0.0

# Detect contradictions (negative counts)
contradictions = sum(
    1 for entity, items in model_state.items()
    for item, count in items.items()
    if count < 0
)
contradiction_rate = contradictions / total_slots if total_slots > 0 else 0.0

# Context loss (missing entities)
expected_entities = set(expected_state.keys())
model_entities = set(model_state.keys())
missing = expected_entities - model_entities
context_loss = len(missing) / len(expected_entities) if expected_entities else 0.0

# Composite score
score = max(0.0, step_accuracy - 0.1 * context_loss - 0.2 * contradiction_rate)

return WorkingMemoryScore(
    probe_id=probe.probe_id,
    step_accuracy=step_accuracy,
    context_loss=context_loss,
    contradiction_rate=contradiction_rate,
    score=score,
    band=probe.band,
    metadata={
        "total_slots": total_slots,
        "correct_slots": correct_slots,
        "missing_entities": list(missing),
        "contradictions": contradictions
    }
)
```

def aggregate_wm_scores(scores: List[WorkingMemoryScore], n_bootstrap: int = 1000) -> Dict[str, Any]:
“”“Aggregate WM scores with bootstrap CI.”””
from neoprompt.metrics.uncertainty import bootstrap_ci

```
if not scores:
    return {"pillar": "wm", "score": 0.0, "ci_95": [0.0, 0.0], "n_probes": 0}

overall_scores = np.array([s.score for s in scores])
overall_mean = np.mean(overall_scores)
ci_lower, ci_upper = bootstrap_ci(overall_scores, n_bootstrap=n_bootstrap)

by_band = {}
for score in scores:
    by_band.setdefault(score.band, []).append(score.score)

return {
    "pillar": "wm",
    "score": overall_mean * 100,
    "ci_95": [ci_lower * 100, ci_upper * 100],
    "step_accuracy": np.mean([s.step_accuracy for s in scores]) * 100,
    "context_loss": np.mean([s.context_loss for s in scores]) * 100,
    "contradiction_rate": np.mean([s.contradiction_rate for s in scores]) * 100,
    "by_band": {b: np.mean(scores_list) * 100 for b, scores_list in by_band.items()},
    "n_probes": len(scores)
}
```

# ============================================================================

# FILE: neoprompt/probes/gf.py (STUB)

# ============================================================================

“””
Abstraction / Fluid Intelligence Probe (Pillar 1)

STUB IMPLEMENTATION - To be completed in Phase 1
“””

def generate_gf_probes(base_case: Dict[str, Any], master_seed: int, n_variants: int = 10) -> List:
“””
Generate Gf Raven matrix probes.

```
TODO: Implement Raven matrix generator with:
- 2×2 to 3×3 grids
- Rule grammar (XOR, progression, rotation, mirror)
- Disjoint symbol alphabets per seed
- Adversarial distractors for band E
"""
raise NotImplementedError("Gf probe generation - implement in Phase 1 Week 1")
```

def score_gf_probe(probe, model_output: Dict[str, Any]) -> Dict[str, Any]:
“””
Score Gf probe with exact match on choice A-H.

```
TODO: Implement exact matching scorer with latency tracking.
"""
raise NotImplementedError("Gf scoring - implement in Phase 1 Week 1")
```

# ============================================================================

# FILE: neoprompt/probes/tool_use.py (STUB)

# ============================================================================

“””
Tool Use Fidelity Probe (Pillar 12)

STUB IMPLEMENTATION - To be completed in Phase 1
“””

def generate_tool_use_probes(base_case: Dict[str, Any], master_seed: int, n_variants: int = 10) -> List:
“””
Generate tool use chain probes.

```
TODO: Implement with:
- Virtual tool server (math.add, math.multiply, search.docs, geo.distance)
- Multi-step tool chains
- Type-checked arguments
- Distractor tools for bands B+
"""
raise NotImplementedError("Tool use probe generation - implement in Phase 1 Week 2")
```

def score_tool_use_probe(probe, model_output: Dict[str, Any]) -> Dict[str, Any]:
“””
Score tool use with:

- Correct call rate
- Argument precision
- Unnecessary call penalty (−0.02 per extra call)

```
TODO: Implement call sequence validation.
"""
raise NotImplementedError("Tool use scoring - implement in Phase 1 Week 2")
```

# ============================================================================

# FILE: neoprompt/probes/grounding.py (STUB)

# ============================================================================

“””
Evidence Grounding & Attribution Probe (Pillar 14)

STUB IMPLEMENTATION - To be completed in Phase 1
“””

def generate_grounding_probes(base_case: Dict[str, Any], master_seed: int, n_variants: int = 10) -> List:
“””
Generate grounding probes with local corpus.

```
TODO: Implement with:
- Local corpus with byte-offset indexing
- Multi-document synthesis tasks
- Span boundary validation
- Adversarial distractor documents
"""
raise NotImplementedError("Grounding probe generation - implement in Phase 1 Week 2")
```

def score_grounding_probe(probe, model_output: Dict[str, Any]) -> Dict[str, Any]:
“””
Score grounding with:

- Precision@3 on span boundaries
- Unsupported claim rate

```
TODO: Implement span validation and claim checking.
"""
raise NotImplementedError("Grounding scoring - implement in Phase 1 Week 2")
```

# ============================================================================

# FILE: neoprompt/probes/robustness.py (STUB)

# ============================================================================

“””
Robustness to Noise/Adversarial Inputs Probe (Pillar 16)

STUB IMPLEMENTATION - To be completed in Phase 1
“””

def generate_robustness_probes(base_case: Dict[str, Any], master_seed: int, n_variants: int = 10) -> List:
“””
Generate robustness probes with noise and jailbreak variants.

```
TODO: Implement with:
- Character-level noise injection (ε ∈ [0, 0.4])
- Policy FSM for jailbreak detection
- Seeded attack bank
"""
raise NotImplementedError("Robustness probe generation - implement in Phase 1 Week 2")
```

def score_robustness_probe(probe, model_output: Dict[str, Any]) -> Dict[str, Any]:
“””
Score robustness with:

- AUC under accuracy-vs-noise curve
- Jailbreak leakage rate

```
TODO: Implement noise tolerance and policy violation detection.
"""
raise NotImplementedError("Robustness scoring - implement in Phase 1 Week 2")
```

# ============================================================================

# FILE: examples/run_evaluation.py

# ============================================================================

“””
Runnable example: Load suite → Expand → Execute → Score → Report

This demonstrates the complete evaluation pipeline using a stub model
that always returns correct answers.
“””

import json
import yaml
from pathlib import Path
from typing import Dict, Any, List

class StubModel:
“”“Stub model that always returns correct answers for testing.”””

```
def generate(self, prompt: str, expected_output: Dict[str, Any]) -> Dict[str, Any]:
    """Return expected output (perfect model)."""
    return expected_output
```

def load_suite(suite_path: str) -> Dict[str, Any]:
“”“Load suite YAML specification.”””
with open(suite_path, ‘r’) as f:
return yaml.safe_load(f)

def expand_probes(suite: Dict[str, Any]) -> List:
“”“Expand base cases into probe variants.”””
from neoprompt.probes.wm import generate_wm_probes

```
all_probes = []
master_seed = suite["master_seed"]
n_variants = suite["settings"]["variants_per_case"]

for pillar in suite["pillars"]:
    if not pillar["enabled"]:
        continue
    
    pillar_id = pillar["pillar_id"]
    
    for family in pillar["families"]:
        for case in family["cases"]:
            # Only WM is implemented for MVP demo
            if pillar_id == "wm":
                probes = generate_wm_probes(case, master_seed, n_variants)
                all_probes.extend(probes)
            else:
                print(f"Skipping {pillar_id} (not implemented in MVP demo)")

return all_probes
```

def execute_probes(probes: List, model: StubModel) -> List[Dict[str, Any]]:
“”“Execute probes against model.”””
artifacts = []

```
for probe in probes:
    # Model generates output
    model_output = model.generate(
        prompt=probe.narrative + "\n\n" + probe.question,
        expected_output={"final_state": probe.expected_output}
    )
    
    artifacts.append({
        "probe_id": probe.probe_id,
        "pillar": probe.pillar,
        "band": probe.band,
        "model_output": model_output,
        "expected_output": probe.expected_output,
        "probe": probe
    })

return artifacts
```

def score_artifacts(artifacts: List[Dict[str, Any]]) -> Dict[str, List]:
“”“Score all artifacts and group by pillar.”””
from neoprompt.probes.wm import score_wm_probe, WorkingMemoryScore

```
scores_by_pillar = {"wm": []}

for artifact in artifacts:
    if artifact["pillar"] == "wm":
        score = score_wm_probe(
            probe=artifact["probe"],
            model_output=artifact["model_output"]
        )
        scores_by_pillar["wm"].append(score)

return scores_by_pillar
```

def aggregate_scores(scores_by_pillar: Dict[str, List]) -> Dict[str, Dict]:
“”“Aggregate scores by pillar.”””
from neoprompt.probes.wm import aggregate_wm_scores

```
aggregated = {}

if "wm" in scores_by_pillar and scores_by_pillar["wm"]:
    aggregated["wm"] = aggregate_wm_scores(scores_by_pillar["wm"])

return aggregated
```

def generate_report(pillar_results: Dict[str, Dict], output_path: str):
“”“Generate JSON report.”””
from datetime import datetime

```
report = {
    "version": "1.0.0",
    "timestamp": datetime.utcnow().isoformat(),
    "pillars": pillar_results,
    "composite": {
        "score": pillar_results.get("wm", {}).get("score", 0.0),
        "weights": {"wm": 1.0},  # Only WM in MVP demo
        "note": "MVP demo - only Working Memory pillar implemented"
    }
}

with open(output_path, 'w') as f:
    json.dump(report, f, indent=2)

print(f"\nReport written to: {output_path}")
print(f"Working Memory Score: {report['pillars']['wm']['score']:.1f}")
print(f"95% CI: [{report['pillars']['wm']['ci_95'][0]:.1f}, {report['pillars']['wm']['ci_95'][1]:.1f}]")
```

def main():
“”“Run complete evaluation pipeline.”””
print(“Cognitive Pillars Evaluation Harness - MVP Demo”)
print(”=” * 60)

```
# 1. Load suite
print("\n1. Loading suite specification...")
suite = load_suite("examples/suite.yml")
print(f"   Loaded: {suite['name']}")
print(f"   Master seed: {suite['master_seed']}")

# 2. Expand probes
print("\n2. Expanding base cases into probe variants...")
probes = expand_probes(suite)
print(f"   Generated {len(probes)} probes")

# 3. Execute
print("\n3. Executing probes against stub model...")
model = StubModel()
artifacts = execute_probes(probes, model)
print(f"   Executed {len(artifacts)} probes")

# 4. Score
print("\n4. Scoring probe responses...")
scores_by_pillar = score_artifacts(artifacts)
print(f"   Scored {sum(len(v) for v in scores_by_pillar.values())} probes")

# 5. Aggregate
print("\n5. Aggregating scores by pillar...")
pillar_results = aggregate_scores(scores_by_pillar)

# 6. Report
print("\n6. Generating report...")
generate_report(pillar_results, "report.dev.json")

print("\n" + "=" * 60)
print("Evaluation complete!")
```

if **name** == “**main**”:
main()

# ============================================================================

# FILE: examples/suite.yml

# ============================================================================

“””
Example suite specification (YAML format)

## Save this as a separate file: examples/suite.yml

version: “1.0.0”
name: “MVP Cognitive Pillars Suite”
description: “Minimum viable product suite for Working Memory pillar”

master_seed: 42

settings:
variants_per_case: 10
band_distribution:
A: 0.2
B: 0.25
C: 0.25
D: 0.2
E: 0.1

pillars:

- pillar_id: “wm”
  enabled: true
  families:
  - family_id: “wm_bags_v1”
    cases:
    - case_id: “wm_001”
      description: “Entity tracking with moves”
    - case_id: “wm_002”
      description: “Entity tracking with contradictions”
- pillar_id: “gf”
  enabled: false
  families:
  - family_id: “gf_raven_v1”
    cases:
    - case_id: “gf_001”
      description: “Raven matrices - single rule”
- pillar_id: “tool_use”
  enabled: false
  families:
  - family_id: “tool_chain_v1”
    cases:
    - case_id: “tool_001”
      description: “Sequential tool calls”
      “””

# ============================================================================

# FILE: neoprompt/schemas/wm_bags_v1.json

# ============================================================================

“””
JSON Schema for Working Memory probe responses

Save this as a separate file: neoprompt/schemas/wm_bags_v1.json
{
“$schema”: “http://json-schema.org/draft-07/schema#”,
“title”: “Working Memory Bags Response”,
“description”: “Final state of entity inventories after moves”,
“type”: “object”,
“required”: [“final_state”],
“properties”: {
“final_state”: {
“type”: “object”,
“description”: “Map of entity names to their final inventories”,
“patternProperties”: {
“^Agent[A-Z]$”: {
“type”: “object”,
“description”: “Entity inventory”,
“patternProperties”: {
“^item_[a-z]$”: {
“type”: “integer”,
“minimum”: 0,
“description”: “Item count (must be non-negative)”
}
},
“additionalProperties”: false
}
},
“additionalProperties”: false
}
},
“additionalProperties”: false
}
“””

# ============================================================================

# TESTS: tests/unit/test_wm_probe.py

# ============================================================================

“””
Unit tests for Working Memory probe

Run with: pytest tests/unit/test_wm_probe.py -v
“””

import pytest
import numpy as np

def test_seed_determinism():
“”“Same seed produces same probes.”””
from neoprompt.probes.wm import generate_wm_probes

```
base_case = {"case_id": "wm_001"}

probes1 = generate_wm_probes(base_case, master_seed=42, n_variants=5)
probes2 = generate_wm_probes(base_case, master_seed=42, n_variants=5)

assert len(probes1) == len(probes2) == 5

for p1, p2 in zip(probes1, probes2):
    assert p1.probe_id == p2.probe_id
    assert p1.seed_hash == p2.seed_hash
    assert p1.narrative == p2.narrative
```

def test_band_distribution():
“”“Probes distributed across bands A-E.”””
from neoprompt.probes.wm import generate_wm_probes
from collections import Counter

```
base_case = {"case_id": "wm_001"}
probes = generate_wm_probes(base_case, master_seed=42, n_variants=10)

bands = [p.band for p in probes]
band_counts = Counter(bands)

assert band_counts["A"] == 2
assert band_counts["B"] == 2
assert band_counts["C"] == 2
assert band_counts["D"] == 2
assert band_counts["E"] == 2
```

def test_perfect_score():
“”“Perfect match scores 1.0.”””
from neoprompt.probes.wm import WorkingMemoryProbe, score_wm_probe

```
probe = WorkingMemoryProbe(
    probe_id="test",
    pillar="wm",
    family="wm_bags_v1",
    band="B",
    seed_hash="hash",
    narrative="...",
    question="...",
    schema_path="...",
    expected_output={"AgentA": {"item_a": 3, "item_b": 1}},
    metadata={}
)

model_output = {"final_state": {"AgentA": {"item_a": 3, "item_b": 1}}}

score = score_wm_probe(probe, model_output)

assert score.step_accuracy == 1.0
assert score.score == 1.0
```

def test_contradiction_penalty():
“”“Negative counts penalized.”””
from neoprompt.probes.wm import WorkingMemoryProbe, score_wm_probe

```
probe = WorkingMemoryProbe(
    probe_id="test",
    pillar="wm",
    family="wm_bags_v1",
    band="B",
    seed_hash="hash",
    narrative="...",
    question="...",
    schema_path="...",
    expected_output={"AgentA": {"item_a": 3}},
    metadata={}
)

model_output = {"final_state": {"AgentA": {"item_a": -5}}}

score = score_wm_probe(probe, model_output)

assert score.contradiction_rate > 0
assert score.score < score.step_accuracy
```

def test_no_network_access():
“”“Verify no network calls during generation.”””
import socket
from neoprompt.probes.wm import generate_wm_probes

```
original_socket = socket.socket

def blocked_socket(*args, **kwargs):
    raise RuntimeError("Network access attempted!")

socket.socket = blocked_socket

try:
    base_case = {"case_id": "wm_001"}
    probes = generate_wm_probes(base_case, master_seed=42, n_variants=2)
    assert len(probes) == 2
finally:
    socket.socket = original_socket
```

# ============================================================================

# USAGE INSTRUCTIONS

# ============================================================================

## “””

SETUP:

1. Create directory structure:
   mkdir -p neoprompt/{probes,generators,metrics,schemas}
   mkdir -p tests/unit
   mkdir -p examples
2. Extract files from this artifact into appropriate locations
3. Install dependencies:
   pip install numpy scipy pyyaml pytest
4. Create examples/suite.yml with the YAML content above
5. Create neoprompt/schemas/wm_bags_v1.json with the schema above

## RUN EVALUATION

python examples/run_evaluation.py

## RUN TESTS

pytest tests/unit/test_wm_probe.py -v

## EXPECTED OUTPUT

# Cognitive Pillars Evaluation Harness - MVP Demo

1. Loading suite specification…
   Loaded: MVP Cognitive Pillars Suite
   Master seed: 42
2. Expanding base cases into probe variants…
   Generated 20 probes
3. Executing probes against stub model…
   Executed 20 probes
4. Scoring probe responses…
   Scored 20 probes
5. Aggregating scores by pillar…
6. Generating report…

Report written to: report.dev.json
Working Memory Score: 100.0
95% CI: [100.0, 100.0]

============================================================
Evaluation complete!

## NEXT STEPS

1. Review report.dev.json for detailed metrics
2. Implement remaining MVP pillars (Gf, Tool Use, Grounding, Robustness)
3. Add real model adapter (replace StubModel)
4. Expand test coverage
5. Follow 6-week roadmap for full 21-pillar implementation
   “””
