import dspy


class HotpotQA(dspy.Module):
    def __init__(self, search, num_docs=3, num_hops=2):
        self.search = search
        self.num_docs, self.num_hops = num_docs, num_hops
        self.generate_query  = dspy.ChainOfThought("question, notes -> search_query")
        self.append_notes    = dspy.ChainOfThought("question, notes, context -> new_notes: list[str]")
        self.generate_answer = dspy.ChainOfThought("question, notes -> answer")

    def forward(self, question: str) -> dspy.Prediction:
        notes        = []
        all_passages = []

        for _ in range(self.num_hops):
            query   = self.generate_query(question=question, notes=notes).search_query
            context = self.search(query, k=self.num_docs)
            all_passages.extend([{"text": t, "score": s} for t, s in context.items()])
            prediction = self.append_notes(question=question, notes=notes, context=context)
            notes.extend(prediction.new_notes)

        pred = self.generate_answer(question=question, notes=notes)

        return dspy.Prediction(
            notes    = notes,
            passages = all_passages,
            answer   = pred.answer,
        )
