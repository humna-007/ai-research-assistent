"""
Dual Groq LLM orchestration:
  Model 1 (openai/gpt-oss-120b) -> Context Summarizer
  Model 2 (openai/gpt-oss-20b)  -> Final Answer Generator
"""

from groq import Groq, APIError, APIConnectionError, RateLimitError, AuthenticationError

SUMMARIZER_MODEL = "openai/gpt-oss-120b"
ANSWER_MODEL = "openai/gpt-oss-20b"


class LLMError(Exception):
    pass


class GroqLLMHandler:
    def __init__(self, api_key: str):
        if not api_key or not api_key.strip():
            raise LLMError("Groq API key is missing.")
        self.client = Groq(api_key=api_key)

    def validate_key(self) -> bool:
        """Lightweight check — lists models to confirm the key works."""
        try:
            self.client.models.list()
            return True
        except AuthenticationError as exc:
            raise LLMError("That API key was rejected by Groq. Please check and try again.") from exc
        except APIConnectionError as exc:
            raise LLMError("Could not reach Groq. Check your internet connection.") from exc
        except APIError as exc:
            raise LLMError(f"Groq API error while validating key: {exc}") from exc

    def _chat(self, model: str, system_prompt: str, user_prompt: str, temperature: float = 0.3, max_tokens: int = 1024) -> str:
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except AuthenticationError as exc:
            raise LLMError("Invalid Groq API key. Please check and re-enter it.") from exc
        except RateLimitError as exc:
            raise LLMError("Groq rate limit reached. Please wait a moment and try again.") from exc
        except APIConnectionError as exc:
            raise LLMError("Could not reach Groq's servers. Check your internet connection.") from exc
        except APIError as exc:
            raise LLMError(f"Groq API error: {exc}") from exc

    def summarize_context(self, retrieved_chunks: list[dict], history_text: str, all_document_names: list[str]) -> str:
        doc_list = ", ".join(all_document_names) if all_document_names else "none"

        if not retrieved_chunks:
            context_block = "No chunks matched this question closely enough to retrieve."
        else:
            lines = []
            for c in retrieved_chunks:
                page = c["metadata"].get("page")
                tag = c["metadata"]["source"] + (f", page {page}" if page else "")
                lines.append(f"[Source: {tag}]\n{c['text']}")
            context_block = "\n\n".join(lines)

        system_prompt = (
            "You are a context summarizer for a research assistant. Condense the "
            "retrieved document excerpts and prior conversation into a compact, "
            "factual summary. Preserve key facts, figures, and source names. "
            "Remove redundancy. Do not answer the question yourself. "
            "You will also be told the full list of documents currently uploaded — "
            "mention if relevant content wasn't found in a specific uploaded document, "
            "rather than implying that document doesn't exist."
        )
        user_prompt = (
            f"ALL DOCUMENTS CURRENTLY UPLOADED: {doc_list}\n\n"
            f"RETRIEVED DOCUMENT EXCERPTS:\n{context_block}\n\n"
            f"PREVIOUS CONVERSATION:\n{history_text}\n\n"
            "Produce a condensed summary for answering the user's next question."
        )
        return self._chat(SUMMARIZER_MODEL, system_prompt, user_prompt)

    def generate_answer(self, question: str, summarized_context: str, all_document_names: list[str]) -> str:
        doc_list = ", ".join(all_document_names) if all_document_names else "none"

        system_prompt = (
            "You are a research assistant that answers using the provided summarized "
            "context, drawn from the user's uploaded documents. If the summarized "
            "context doesn't contain enough information for this specific question, "
            "say so plainly — but never claim a document doesn't exist if it's listed "
            "as uploaded; instead say its content didn't match this particular question. "
            "Mention source documents when relevant. "
            "Respond directly with the answer content only — do not prefix your reply "
            "with headings or labels like 'Answer:', 'Response:', or similar."
        )
        user_prompt = (
            f"DOCUMENTS UPLOADED IN THIS SESSION: {doc_list}\n\n"
            f"SUMMARIZED CONTEXT:\n{summarized_context}\n\n"
            f"QUESTION:\n{question}\n\n"
            "Provide a clear, accurate, well-structured answer."
        )
        return self._chat(ANSWER_MODEL, system_prompt, user_prompt, temperature=0.4, max_tokens=2048)

    def generate_title(self, question: str, answer: str) -> str:
        """Short, memorable chat title — similar to how Claude/ChatGPT title conversations."""
        system_prompt = (
            "Generate a short chat title, 3 to 5 words, summarizing the topic of this "
            "exchange. No quotes, no punctuation at the end, no the words 'chat' or 'title'. "
            "Just the title text."
        )
        user_prompt = f"Question: {question}\n\nAnswer: {answer[:400]}"
        try:
            title = self._chat(ANSWER_MODEL, system_prompt, user_prompt, temperature=0.3)
            title = title.strip().strip('"').strip("'")
            return title[:50] if title else question[:40]
        except LLMError:
            return question[:40] + ("..." if len(question) > 40 else "")