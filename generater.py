"""
Peptide sequence generator using BLOSUM62-based substitutions.

Template format (compatible with your original intent):
- Text inside double quotes "..." is treated as FIXED (kept as-is).
- Text outside quotes is treated as VARIABLE and can be mutated position-wise.

Example:
    template = 'AAA"GGG"TT'
    - VARIABLE: AAA
    - FIXED:    GGG
    - VARIABLE: TT
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Protocol, Sequence, Set, Tuple

import numpy as np

# NOTE:
# Bio.SubsMat is considered legacy in newer Biopython versions.
# Keeping it since your script uses it, but if you upgrade Biopython,
# you may need to switch to Bio.Align.substitution_matrices.
from Bio.SubsMat import MatrixInfo


AMINO_ACIDS: str = "ACDEFGHIKLMNPQRSTVWY"
BLOSUM62: Dict[Tuple[str, str], int] = MatrixInfo.blosum62


# ---------- Domain model (SRP) ----------

@dataclass(frozen=True)
class TemplatePart:
    """
    Represents one contiguous part of the template.

    kind:
        - "fixed"    -> appended as-is
        - "variable" -> each character can be substituted
    """
    kind: str
    content: str


# ---------- Interfaces / Abstractions (DIP + ISP) ----------

class SubstitutionStrategy(Protocol):
    """
    Strategy interface: decides how to substitute one amino acid.
    Allows Open/Closed: you can add new strategies without touching the generator.
    """

    def choose_substitution(self, aa: str) -> str:
        """Return a substituted amino acid (single-letter)."""


# ---------- Concrete Strategies (OCP + LSP) ----------

class Blosum62WeightedStrategy:
    """
    Substitution strategy based on BLOSUM62 scores.

    - Collects all BLOSUM pairs involving 'aa'.
    - Uses weights derived from scores:
        weight = max(score, min_weight)
      Then normalized to probabilities.
    - If no BLOSUM entries exist for aa, falls back to uniform over all amino acids.
    """

    def __init__(
        self,
        blosum: Dict[Tuple[str, str], int] = BLOSUM62,
        amino_acids: str = AMINO_ACIDS,
        min_weight: float = 0.1,
    ) -> None:
        self._blosum = blosum
        self._amino_acids = list(amino_acids)
        self._aa_set = set(amino_acids)
        self._min_weight = float(min_weight)

    def choose_substitution(self, aa: str) -> str:
        scores = self._scores_for(aa)

        # Fallback: if the matrix has no information about aa,
        # we choose uniformly among standard amino acids.
        if not scores:
            return random.choice(self._amino_acids)

        aa_list = list(scores.keys())

        # Ensure all weights are positive to avoid zero-probability issues.
        weights = np.array([max(scores[a], self._min_weight) for a in aa_list], dtype=float)
        weights = weights / weights.sum()

        return random.choices(aa_list, weights=weights, k=1)[0]

    def positive_substitutions(self, aa: str) -> Set[str]:
        """
        Helper for estimating number of variants:
        returns set of substitutions with score > 0.
        """
        scores = self._scores_for(aa)
        return {a for a, s in scores.items() if s > 0}

    def _scores_for(self, aa: str) -> Dict[str, int]:
        """
        Extract all substitution scores for aa from the symmetric matrix dict.
        """
        scores: Dict[str, int] = {}

        for (a1, a2), score in self._blosum.items():
            if a1 == aa and a2 in self._aa_set:
                scores[a2] = score
            elif a2 == aa and a1 in self._aa_set:
                scores[a1] = score

        return scores


# ---------- Parser (SRP) ----------

class TemplateParser:
    """
    Parses the template into fixed/variable parts.

    Convention:
    - inside quotes  => fixed
    - outside quotes => variable
    """

    def parse(self, template: str) -> List[TemplatePart]:
        parts: List[TemplatePart] = []
        buffer: List[str] = []
        inside_quotes = False

        for ch in template:
            if ch == '"':
                # Flush current buffer into a part BEFORE toggling.
                if buffer:
                    parts.append(
                        TemplatePart(
                            kind="fixed" if inside_quotes else "variable",
                            content="".join(buffer),
                        )
                    )
                    buffer = []

                inside_quotes = not inside_quotes
            else:
                buffer.append(ch)

        # Flush tail
        if buffer:
            parts.append(
                TemplatePart(
                    kind="fixed" if inside_quotes else "variable",
                    content="".join(buffer),
                )
            )

        return parts


# ---------- Generator (SRP + DIP) ----------

class SequenceGenerator:
    """
    Generates sequences from a parsed template using an injected substitution strategy.
    """

    def __init__(self, strategy: SubstitutionStrategy) -> None:
        self._strategy = strategy

    def generate_one(self, parts: Sequence[TemplatePart]) -> str:
        seq_chars: List[str] = []

        for part in parts:
            if part.kind == "fixed":
                seq_chars.append(part.content)
            elif part.kind == "variable":
                for aa in part.content:
                    seq_chars.append(self._strategy.choose_substitution(aa))
            else:
                raise ValueError(f"Unknown template part kind: {part.kind}")

        return "".join(seq_chars)

    def generate_unique(self, parts: Sequence[TemplatePart], n: int, max_attempts: int = 100_000) -> List[str]:
        """
        Generate up to n UNIQUE sequences.

        max_attempts prevents an infinite loop if n exceeds the reachable number
        of unique sequences for the given template and strategy.
        """
        sequences: Set[str] = set()
        attempts = 0

        while len(sequences) < n and attempts < max_attempts:
            sequences.add(self.generate_one(parts))
            attempts += 1

        return list(sequences)


# ---------- Utilities (kept separate) ----------

def estimate_max_variants(template: str, strategy: Blosum62WeightedStrategy | None = None) -> int:
    """
    Rough upper bound estimate of how many variants are possible if we only allow
    substitutions with BLOSUM score > 0 at each variable position.

    Note:
    - This is a simplistic multiplicative estimate and assumes independence per position.
    - If a position has no positive substitutions, we default to 20 possibilities.
    """
    parser = TemplateParser()
    parts = parser.parse(template)

    if strategy is None:
        strategy = Blosum62WeightedStrategy()

    total = 1

    for part in parts:
        if part.kind != "variable":
            continue

        for aa in part.content:
            options = strategy.positive_substitutions(aa)
            total *= (len(options) if options else len(AMINO_ACIDS))

    return total


def generate_sequences(template: str, n: int) -> List[str]:
    """
    Convenience function matching your original API.
    """
    parser = TemplateParser()
    parts = parser.parse(template)

    strategy = Blosum62WeightedStrategy()
    generator = SequenceGenerator(strategy)

    return generator.generate_unique(parts, n=n)
