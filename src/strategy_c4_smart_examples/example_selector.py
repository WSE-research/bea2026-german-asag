"""
Smart example selection combining boundary-focused and similarity-based retrieval.

Two modes:
1. BOUNDARY: Select examples near the scoring boundaries (hardest cases)
2. SIMILAR: For each test answer, retrieve most similar training answers per label

Combined: Use boundary examples as fixed anchors + similarity-based examples as dynamic context.
"""

import logging
import random

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

LABEL_ORDER = ["Correct", "Partially correct", "Incorrect"]


class SmartExampleSelector:
    def __init__(self, train_data: list[dict], seed: int = 42):
        """
        Initialize with training data. Pre-computes:
        - Per-question TF-IDF matrix for similarity retrieval
        - Per-question boundary examples (answers closest to decision boundaries)
        """
        self.seed = seed
        self.rng = random.Random(seed)

        # Group by question
        self.by_question = {}  # qid -> {label -> [samples]}
        for s in train_data:
            qid = s["question_id"]
            label = s["score"]
            self.by_question.setdefault(qid, {}).setdefault(label, []).append(s)

        # Pre-compute TF-IDF per question
        self.tfidf_data = {}  # qid -> {"vectorizer": ..., "matrix": ..., "samples": [...]}
        for qid, label_groups in self.by_question.items():
            all_samples = []
            for label in LABEL_ORDER:
                all_samples.extend(label_groups.get(label, []))
            if len(all_samples) < 3:
                continue
            texts = [s["answer"] for s in all_samples]
            vectorizer = TfidfVectorizer(max_features=5000, stop_words=None)  # German, no stop words list
            try:
                matrix = vectorizer.fit_transform(texts)
                self.tfidf_data[qid] = {"vectorizer": vectorizer, "matrix": matrix, "samples": all_samples}
            except ValueError:
                # Empty vocabulary
                continue

        # Pre-compute boundary examples per question
        self.boundary_examples = {}  # qid -> list of boundary examples
        self._compute_boundary_examples()

        logger.info(
            "SmartExampleSelector: %d questions indexed, %d with TF-IDF",
            len(self.by_question),
            len(self.tfidf_data),
        )

    def _compute_boundary_examples(self):
        """
        For each question, find boundary examples: answers that are closest
        to the decision boundary between adjacent score levels.

        Method: Compute TF-IDF similarity between all answer pairs across
        adjacent labels. The most similar cross-label pairs are the boundary cases.
        """
        for qid, tfidf_info in self.tfidf_data.items():
            samples = tfidf_info["samples"]
            matrix = tfidf_info["matrix"]

            # Index samples by position in matrix
            idx_by_label = {}
            for i, s in enumerate(samples):
                idx_by_label.setdefault(s["score"], []).append(i)

            boundary = []

            # Find boundary pairs for each adjacent label pair
            for label_a, label_b in [("Correct", "Partially correct"), ("Partially correct", "Incorrect")]:
                idxs_a = idx_by_label.get(label_a, [])
                idxs_b = idx_by_label.get(label_b, [])

                if not idxs_a or not idxs_b:
                    continue

                # Compute cross-label similarities
                sim = cosine_similarity(matrix[idxs_a], matrix[idxs_b])

                # Find the most similar cross-label pair (highest confusion potential)
                max_idx = np.unravel_index(sim.argmax(), sim.shape)
                best_a = idxs_a[max_idx[0]]
                best_b = idxs_b[max_idx[1]]

                boundary.append({
                    "id": samples[best_a]["id"],
                    "answer": samples[best_a]["answer"],
                    "score": samples[best_a]["score"],
                    "type": "boundary",
                })
                boundary.append({
                    "id": samples[best_b]["id"],
                    "answer": samples[best_b]["answer"],
                    "score": samples[best_b]["score"],
                    "type": "boundary",
                })

            self.boundary_examples[qid] = boundary

    def get_examples(self, sample: dict, n_boundary: int = 2, n_similar: int = 1) -> list[dict]:
        """
        Get examples for scoring a sample. Combines:
        - n_boundary boundary examples per adjacent boundary (up to 2*n_boundary total from boundaries)
        - n_similar similar examples per label retrieved via TF-IDF

        Excludes the sample itself (by ID) to prevent data leakage.

        Returns list of {"id", "answer", "score"} dicts, sorted by label order.
        """
        qid = sample["question_id"]
        sample_id = sample["id"]

        examples = []
        seen_ids = {sample_id}  # exclude self

        # 1. Add boundary examples
        for ex in self.boundary_examples.get(qid, []):
            if ex["id"] not in seen_ids:
                examples.append({"id": ex["id"], "answer": ex["answer"], "score": ex["score"]})
                seen_ids.add(ex["id"])

        # 2. Add similarity-based examples per label
        if qid in self.tfidf_data:
            tfidf_info = self.tfidf_data[qid]
            vectorizer = tfidf_info["vectorizer"]
            matrix = tfidf_info["matrix"]
            all_samples = tfidf_info["samples"]

            try:
                query_vec = vectorizer.transform([sample["answer"]])
                similarities = cosine_similarity(query_vec, matrix)[0]

                for label in LABEL_ORDER:
                    # Get indices for this label, excluding already-selected
                    candidates = [
                        (i, similarities[i])
                        for i, s in enumerate(all_samples)
                        if s["score"] == label and s["id"] not in seen_ids
                    ]
                    candidates.sort(key=lambda x: x[1], reverse=True)

                    for idx, sim_score in candidates[:n_similar]:
                        s = all_samples[idx]
                        examples.append({"id": s["id"], "answer": s["answer"], "score": s["score"]})
                        seen_ids.add(s["id"])
            except ValueError:
                pass

        # 3. Fill remaining slots with random if needed (ensure at least 1 per label)
        for label in LABEL_ORDER:
            if not any(ex["score"] == label for ex in examples):
                candidates = [
                    s
                    for s in self.by_question.get(qid, {}).get(label, [])
                    if s["id"] not in seen_ids
                ]
                if candidates:
                    pick = self.rng.choice(candidates)
                    examples.append({"id": pick["id"], "answer": pick["answer"], "score": pick["score"]})
                    seen_ids.add(pick["id"])

        # Sort by label order
        label_rank = {l: i for i, l in enumerate(LABEL_ORDER)}
        examples.sort(key=lambda ex: label_rank.get(ex["score"], 99))

        return examples
