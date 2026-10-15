"""Word-level transcript alignment and pronunciation feedback."""

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List


_WORD_PATTERN = re.compile(r"\b[\w']+\b", re.UNICODE)


@dataclass(frozen=True)
class WordAssessment:
    word: str
    color: str
    similarity: float


@dataclass(frozen=True)
class PronunciationResult:
    reference: str
    transcription: str
    accuracy: int
    words: List[WordAssessment]


def assess_pronunciation(reference: str, transcription: str) -> PronunciationResult:
    """Align recognized words with the reference and return word-level feedback."""
    reference_words = _WORD_PATTERN.findall(reference)
    spoken_words = _WORD_PATTERN.findall(transcription)
    if not reference_words:
        raise ValueError("The reference text does not contain any words.")

    expected = [word.casefold() for word in reference_words]
    spoken = [word.casefold() for word in spoken_words]

    rows = len(expected) + 1
    columns = len(spoken) + 1
    costs = [[0] * columns for _ in range(rows)]
    for i in range(rows):
        costs[i][0] = i
    for j in range(columns):
        costs[0][j] = j

    for i in range(1, rows):
        for j in range(1, columns):
            substitution_cost = 0 if expected[i - 1] == spoken[j - 1] else 1
            costs[i][j] = min(
                costs[i - 1][j] + 1,
                costs[i][j - 1] + 1,
                costs[i - 1][j - 1] + substitution_cost,
            )

    aligned = [WordAssessment(word, "red", 0.0) for word in reference_words]
    substitutions = 0
    deletions = 0
    insertions = 0
    i, j = len(expected), len(spoken)
    while i or j:
        if i and j and expected[i - 1] == spoken[j - 1] and costs[i][j] == costs[i - 1][j - 1]:
            aligned[i - 1] = WordAssessment(reference_words[i - 1], "green", 1.0)
            i -= 1
            j -= 1
            continue

        if i and j and costs[i][j] == costs[i - 1][j - 1] + 1:
            similarity = SequenceMatcher(None, expected[i - 1], spoken[j - 1]).ratio()
            aligned[i - 1] = WordAssessment(
                reference_words[i - 1],
                "yellow" if similarity >= 0.6 else "red",
                similarity,
            )
            substitutions += 1
            i -= 1
            j -= 1
        elif i and costs[i][j] == costs[i - 1][j] + 1:
            deletions += 1
            i -= 1
        else:
            insertions += 1
            j -= 1

    error_rate = (substitutions + deletions + insertions) / len(expected)
    accuracy = round(max(0.0, 1.0 - error_rate) * 100)
    return PronunciationResult(reference, transcription, accuracy, aligned)
