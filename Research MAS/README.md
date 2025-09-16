# 📚 Research & Content Summarization MAS

### Concept

Building a **multi-agent system** where a *Supervisor Agent* orchestrates research on a topic. It decides which sub-agent(s) to call:

* A **Web Research Agent** that can scrape/search content.
* A **Summarizer Agent** that can compress long text into concise notes.
* A **Fact-Checker Agent** that verifies claims.
* A **Citation Agent** that formats references (APA/MLA).

Instead of being linear, the supervisor chooses paths dynamically:

* If research content is too long → pass to Summarizer.
* If claims are detected → route to Fact-Checker.
* If sources are valid → route to Citation Agent.

This creates a **hierarchical + branching workflow**.

---

### Workflow (Non-Linear)

```
Supervisor
  ├──> Web Research Agent → Summarizer (if too long)
  ├──> Fact-Checker Agent (if claims detected)
  └──> Citation Agent (if source used in report)
```

So the flow could be:

* Supervisor → Research → Summarizer → Supervisor
* Supervisor → Research → Fact-Checker → Supervisor
* Supervisor → Citation Agent → Supervisor

The supervisor loops until it has enough material for a final report.

---

### Example Tools

Each agent gets **multiple tools** via `bind_tools`, so it must pick which to call:

#### Web Research Agent tools:

* `search_web(query: str)`
* `fetch_url(url: str)`

#### Summarizer Agent tools:

* `summarize_text(text: str, style: str = "bullet")`
* `extract_keywords(text: str, n: int = 10)`

#### Fact-Checker Agent tools:

* `search_claim(claim: str)`
* `check_against_database(claim: str)`

#### Citation Agent tools:

* `format_citation(source: dict, style: str)`

---

### Hierarchical Control

* **Supervisor**: Always at the top, orchestrates.
* **Workers**: Researcher, Summarizer, Fact-Checker, Citation.
* Supervisor can call multiple workers in *parallel* or *conditionally*, making the system agentic and non-linear.

---

### Why it’s a Good Practice MAS

* Non-linear branching (Supervisor decides path).
* Multiple tools per agent (bind\_tools makes sense).
* Hierarchical structure (Supervisor at top, workers below).
* Real-world applicable (automated research assistant).
