import dspy
from enum import Enum


class ClaimVerificationLabel(Enum):
    SUPPORTED     = "SUPPORTED"
    NOT_SUPPORTED = "NOT_SUPPORTED"


class _HoverAnswerSignature(dspy.Signature):
    """Verify whether the claim is supported or refuted based on the collected notes."""

    claim: str       = dspy.InputField()
    notes: list[str] = dspy.InputField()
    label: ClaimVerificationLabel = dspy.OutputField()


class HoVer(dspy.Module):
    def __init__(self, search, num_docs=3, num_hops=2):
        self.search = search
        self.num_docs, self.num_hops = num_docs, num_hops
        self.generate_query  = dspy.ChainOfThought("claim, notes -> search_query")
        self.append_notes    = dspy.ChainOfThought("claim, notes, context -> new_notes: list[str]")
        self.generate_answer = dspy.ChainOfThought(_HoverAnswerSignature)

    def forward(self, claim: str) -> dspy.Prediction:
        notes        = []
        all_passages = []

        for _ in range(self.num_hops):
            query   = self.generate_query(claim=claim, notes=notes).search_query
            context = self.search(query, k=self.num_docs)
            all_passages.extend([{"text": t, "score": s} for t, s in context.items()])
            prediction = self.append_notes(claim=claim, notes=notes, context=context)
            notes.extend(prediction.new_notes)

        pred = self.generate_answer(claim=claim, notes=notes)

        return dspy.Prediction(
            notes     = notes,
            passages  = all_passages,
            label     = pred.label,
            label_int = int(pred.label == ClaimVerificationLabel.SUPPORTED),
        )
