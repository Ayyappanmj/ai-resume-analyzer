# Interview questions this project could prompt (and how to answer them)

**System design / architecture**
1. Walk me through the architecture of your resume analyzer — why FastAPI + Streamlit instead
   of a single framework?
2. How would this scale to thousands of concurrent resume uploads?
3. Why did you choose Sentence-Transformers over a simple keyword-matching approach for
   similarity — and why keep the lexical fallback at all?

**NLP / ML**
4. How does your skill-extraction avoid false positives on ambiguous words (e.g. "R", "Go")?
5. How is the ATS score weighted, and how did you decide on those weights?
6. What are the limitations of using an LLM (Llama 3.1) to rewrite resume bullets, and how do
   you prevent it from fabricating experience?

**Engineering practices**
7. How do you handle the case where Ollama or the embedding model is unavailable in production?
8. What's in your CI pipeline, and why do the tests mock the LLM and embedding calls?
9. How would you add authentication and multi-tenant history to this app?

**Product**
10. What would you build next if you had another two weeks on this project?
