# 📊 Financial Fraud Detection MAS

### 🎯 Goal:

Build a system to analyze financial transactions, detect fraud, classify risks, and produce a compliance-ready report. The workflow is **non-linear**, **hierarchical**, and requires agents to pick from multiple tools.

---

## 🧩 Agents & Workflow

### 1. **Ingestion Agent (Recon-like)**

* **Role:** Fetch and normalize transaction data from multiple sources (CSV, API, SQL).
* **Tools it can pick from:**

  * `load_csv`
  * `fetch_api`
  * `query_database`
* **Behavior:** It loops until it has a *complete dataset* (just like your recon agent loops until the network map is complete).
* **Output:** A structured dataset (transactions, users, metadata).

---

### 2. **Splits into Two Branches:**

#### A. **Transaction Analysis Agent (Recon Analysis)**

* **Role:** Parse the dataset for trends, anomalies, and patterns (e.g., spikes in amounts, unusual geographies).
* **Tools:**

  * `statistical_summary`
  * `time_series_analysis`
  * `outlier_detection`
* **Output:** A readable analysis report (tables, graphs, summaries).

#### B. **Fraud Detection Agent (Vulnerability Agent)**

* **Role:** Detect potential fraud by applying ML or rule-based checks.
* **Tools:**

  * `ml_fraud_model`
  * `rule_based_checker`
  * `graph_based_detection` (to find fraud rings / collusion)
* **Output:** A list of suspicious transactions, enriched with fraud indicators.

---

### 3. **Risk Classification Agent (CVSS Classifier)**

* **Role:** For each suspicious transaction, assign a **risk score** (Low/Medium/High) similar to CVSS scoring.
* **Tools:**

  * `risk_scorer`
  * `regulatory_checker`
* **Output:** A structured risk report: `Transaction_ID → Risk_Score → Reason`.

---

### 4. **Supervisor / Summary Agent**

* **Inputs:**

  * Analysis report (from Transaction Analysis Agent)
  * Risk report (from Risk Classifier Agent)
* **Role:** Combine both into a **Fraud & Risk Report** that can be exported.
* **Outputs:**

  * PDF Compliance Report
  * JSON Dashboard Data

---

## 🔄 Hierarchical + Non-linear Flow

* **Hierarchical:** Supervisor at the top, ingestion at the bottom, branching in the middle.
* **Non-linear:** After ingestion, the system forks → one branch for *statistical analysis*, another for *fraud detection*. They only converge at the supervisor.
* **Tool-selective:** Each agent has multiple tools to choose from (bind\_tools fits perfectly).
* **Looping:** Ingestion agent loops until the dataset is fully acquired and cleaned.

---

## 🔧 Analogy to Your Network MAS

* **Recon Agent → Ingestion Agent**
* **Recon Analysis Agent → Transaction Analysis Agent**
* **Vulnerability Agent → Fraud Detection Agent**
* **CVSS Classifier Agent → Risk Classification Agent**
* **Supervisor Agent → Supervisor Agent**

Both produce a **final report + dashboard output**, but in *different domains*.

---

👉 This way, you’ll practice:

* Multi-branch workflows
* Hierarchical orchestration
* Tool selection with `bind_tools`
* Loops until completion

### FRAUD SIGNALS
⚠️ Fraud signals included from the data.

Bank: duplicate high withdrawals (BTX1007 & BTX1008), sudden international location (Lagos).

Credit Card: repeated luxury purchases in Dubai, “TestMerchant” with Unknown location.

Online Payments: duplicate payments from same IP, large payment from suspicious IP.
