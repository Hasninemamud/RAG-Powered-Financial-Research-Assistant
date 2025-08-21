from .retriever import Retriever
from .memory import ConversationMemory
from .llm import call_llm   # new import


class Chatbot:
    def __init__(self):
        self.retriever = Retriever()
        self.memory = ConversationMemory()

    def answer(self, session_id: str, question: str, top_k: int = 3, use_llm: bool = False) -> str:
        q_ctx = self.memory.contextualize(session_id, question)
        results = self.retriever.search(q_ctx, top_k=top_k)
        topic = self.memory.infer_topic(question)
        self.memory.update(session_id, question, topic)

        if not results:
            return "I couldn't find anything relevant in the document."

        if use_llm:
            # ✅ Summarized answer
            summary = call_llm(question, results)
            return f"Q: {question}\n\n{summary}\n\n---\nSources:\n" + \
                   "\n".join([f"- Page {r['page']}, score {r['score']:.3f}" for r in results])
        else:
            # ✅ Extractive fallback (default)
            lines = [f"Q: {question}", "", "Top matches:"]
            for r in results:
                snippet = r["text"].strip().replace("\n", " ")
                if len(snippet) > 400:
                    snippet = snippet[:400] + "…"
                lines.append(f"- (page {r['page']}, score {r['score']:.3f}) {snippet}")
            return "\n".join(lines)


def chat_loop(session_id: str, use_llm: bool = False):
    bot = Chatbot()
    print("Policy Chatbot. Type /exit to quit.")
    while True:
        q = input("You: ").strip()
        if q.lower() in {"/exit", "exit", "quit"}:
            break
        print()
        print(bot.answer(session_id, q, use_llm=use_llm))
        print()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Chat with the policy chatbot (CLI).")
    parser.add_argument("--session", default="default", help="Session ID (memory is per-session)")
    parser.add_argument("--top_k", type=int, default=3, help="Number of passages to return")
    parser.add_argument("--llm", action="store_true", help="Use LLM for summarization")
    args = parser.parse_args()
    chat_loop(args.session, use_llm=args.llm)


if __name__ == "__main__":
    main()
