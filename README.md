# ApplyX

### Autonomous Application Agent for Jobs, Internships & Scholarships

> **ApplyX turns an opportunity into an evidence-backed application decision.**

ApplyX is an AI application agent that understands opportunity requirements, verifies candidate eligibility, reasons over uncertainty, prepares applications, and keeps the human in control before any real-world submission.

---

## 🎥 Demo

**Demo Video:**  
https://youtu.be/nJQpALBpfhA?si=0mVUAtGo0Yq3wf-e
---

## 🚨 Problem

Applying for jobs, internships, and scholarships is repetitive and time-consuming.

Candidates often have to:

- Read and understand requirements manually
- Check whether they are eligible
- Compare every requirement with their profile
- Handle missing or uncertain information
- Fill similar forms repeatedly
- Track application states manually
- Decide whether an application is worth pursuing

Existing form-filling automation mainly focuses on typing information into forms.

**The missing layer is application decisioning.**

---

## 💡 Solution

ApplyX acts as an AI application agent that works through the decision process before submission.

It:

- Understands and normalizes opportunity requirements
- Checks for duplicate applications
- Matches requirements against candidate information
- Evaluates eligibility requirement-by-requirement
- Identifies uncertainty instead of blindly assuming eligibility
- Asks the user for clarification when required
- Re-evaluates the application after new information is provided
- Makes an explicit **APPLY / REVIEW / SKIP** decision
- Generates an application draft
- Waits for human approval
- Submits only an explicitly approved version
- Maintains a traceable application lifecycle

---

## 🔄 How ApplyX Works

### 1. Opportunity Understanding

The opportunity is received and its requirements are normalized into structured information.

### 2. Duplicate Protection

ApplyX checks whether the same user has already created an application for the same opportunity.

The database also enforces:

```text
UNIQUE(user_id, opportunity_id)
This provides authoritative duplicate protection.
3. Candidate Matching
Each requirement is evaluated against the candidate profile and available evidence.
Examples include:
Education
Graduation year
Skills
Experience
Enrollment status
Background-check requirements
Other opportunity-specific criteria
4. Eligibility Decision
ApplyX produces an explicit decision:
APPLY
REVIEW
SKIP
The decision is produced through a structured decision tool rather than by parsing free-form AI text.
5. Clarification & Re-evaluation
If important information is uncertain, ApplyX asks the user for clarification.
The process becomes:
Uncertainty
    ↓
User Clarification
    ↓
Knowledge Base Update
    ↓
Re-evaluation
    ↓
APPLY / REVIEW / SKIP
Only the permitted knowledge-base field is updated, and the clarification is recorded for traceability.
6. Application Draft
If the decision is APPLY, ApplyX prepares an application draft.
The draft is not automatically submitted.
7. Human Authorization
The user reviews the generated draft and explicitly approves a specific version.
AI Reasoning
     ↓
Draft V1
     ↓
Human Approval
     ↓
Approved Version
     ↓
Submission Policy Gate
     ↓
Browser Submission
8. Submission
Only an explicitly approved draft version can cross the submission boundary.
This separates AI reasoning from real-world action.
🤖 Agentic Architecture
ApplyX uses Strands Agents for the agentic reasoning loop.
The application lifecycle is controlled by a deterministic orchestration layer.
User / Opportunity
        ↓
Application Orchestrator
        ↓
Normalize Opportunity
        ↓
Duplicate Check
        ↓
Create Application
        ↓
Strands Agent Run #1
        ↓
Requirement Matching
        ↓
Eligibility Decision
        ↓
 ┌────────────┬────────────┬────────────┐
 │            │            │
APPLY       REVIEW        SKIP
 │            │            │
Draft       Clarification  End
 │            │
Human       User Answer
Approval      │
 │         KB Update
 │            │
 │       Strands Run #2
 │            │
 │       Re-evaluation
 │            │
 └────────────┴─────────────
              ↓
      Human Authorization
              ↓
   Deterministic Submission
              ↓
          Playwright
              ↓
          Confirmation
Core Principle
Autonomous in reasoning. Deterministic in control. Human in authorization.
The AI agent can reason and prepare an application, but it cannot independently control the complete application lifecycle.
🧠 Why This Architecture?
ApplyX separates three responsibilities:
AI Reasoning
Strands Agents handle:
Opportunity understanding
Requirement interpretation
Candidate-context reasoning
Uncertainty identification
Clarification requests
Application drafting
Deterministic Control
The orchestration layer handles:
Application state transitions
Duplicate protection
Knowledge-base validation
Decision enforcement
Draft versioning
Approval validation
Submission policy checks
Browser workflow control
Human Authorization
The user controls the irreversible step:
Draft
  ↓
Human Review
  ↓
Human Approval
  ↓
Submission
This prevents an AI-generated decision from directly becoming a real-world application submission.
🔁 Application Lifecycle
ApplyX maintains an explicit application state machine.
Typical lifecycle:
DISCOVERED
    ↓
NORMALIZED
    ↓
DUPLICATE_CHECKED
    ↓
ANALYZED
    ↓
APPLY / REVIEW_REQUIRED / SKIPPED
    ↓
DRAFT_CREATED
    ↓
AWAITING_APPROVAL
    ↓
APPROVED
    ↓
SUBMITTING
    ↓
SUBMITTED
When clarification is required:
REVIEW_REQUIRED
      ↓
CLARIFICATION_REQUIRED
      ↓
RE_EVALUATING
      ↓
APPLY / REVIEW_REQUIRED / SKIPPED
Failures are explicitly represented rather than silently treated as success.
🛡️ Human Control & Safety
ApplyX never turns an AI decision directly into an irreversible submission.
The submission boundary requires:
A valid application
A valid application state
A generated draft
An explicitly approved draft version
A successful submission policy check
Controlled browser execution
The approved draft version is bound to the submission.
If the draft is edited after approval, the previous approval is no longer valid.
🔍 Traceability
Application actions are recorded as structured trace events.
The trace can represent events such as:
Opportunity Normalized
        ↓
Duplicate Checked
        ↓
Candidate Matched
        ↓
Eligibility Decided
        ↓
Draft Created
        ↓
Human Approval
        ↓
Policy Gate
        ↓
Browser Submission
        ↓
Confirmation
The system distinguishes different agent runs, including:
RUN-1
RUN-2
This makes the clarification and re-evaluation loop auditable.
🧰 Tech Stack
Layer
Technology
Frontend
Streamlit
Agent Framework
Strands Agents
Language
Python
AI Model
Gemini
Backend Logic
Python
Database
SQLite
Browser Automation
Playwright
State Management
Deterministic Application State Machine
Version Control
Git + GitHub
📁 Project Structure
ApplyX/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── core/
│   ├── agent_tools.py
│   ├── decision_engine.py
│   ├── matching_engine.py
│   ├── orchestrator.py
│   ├── schemas.py
│   ├── state_machine.py
│   └── strands_agent.py
│
├── diagnostics/
│   └── run_benchmark.py
│
├── tests/
│   └── test_core.py
│
├── data/
│   └── evaluation/
│
└── docs/
    └── architecture.png
⚙️ Local Setup
1. Clone the repository
git clone https://github.com/harshvpatil03-arch/ApplyX.git
cd ApplyX
2. Create a virtual environment
Windows PowerShell:
python -m venv .venv
.venv\Scripts\Activate.ps1
3. Install dependencies
pip install -r requirements.txt
4. Configure the API key
Create a local .env file:
GEMINI_API_KEY=your_api_key_here
Do not commit .env or API keys to GitHub.
The repository .gitignore is configured to exclude environment files and local database files.
5. Run ApplyX
streamlit run app.py
The application will open in your browser.
🧪 Evaluation & Reliability
ApplyX includes evaluation and test infrastructure for important application-control scenarios.
The system is designed to verify:
Duplicate application protection
Requirement matching
Eligibility decisions
Clarification handling
Knowledge-base updates
Draft versioning
Human approval
Submission policy enforcement
Browser failure handling
Application state transitions
Agent tool budgets
Traceability across agent runs
The goal is not only to demonstrate an AI agent, but also to demonstrate controlled and reliable agent behavior.
🚧 Current Limitations
The current hackathon version has several limitations:
Opportunity discovery is not fully implemented
Persistence currently uses SQLite
Different application websites require different automation handling
Not every application form can currently be automated
More comprehensive evaluation datasets are required
Authentication, CAPTCHA, and anti-bot flows require additional handling
The current browser workflow is designed for controlled demonstration
These limitations define the path toward a production-scale system.
🚀 Roadmap
1. Opportunity Discovery
Automatically discover relevant jobs, internships, and scholarships.
2. Application-Site Adapters
Support more application websites and form structures.
3. Scalable Infrastructure
Move from local SQLite persistence toward a distributed production database.
4. Stronger Evaluation
Expand the benchmark with larger and more diverse real-world scenarios.
5. Better Document Understanding
Improve extraction and verification of candidate documents and supporting evidence.
6. Production Authentication
Add secure authentication and controlled handling of CAPTCHA and anti-bot workflows.
7. Scalable Application Automation
Expand controlled application automation while maintaining human authorization.
This hackathon version proves the decision loop. The next step is production-scale automation.
🎯 Why ApplyX?
Most application automation focuses on:
"Fill the form."
ApplyX focuses on:
"Should we apply?"
        ↓
"Why?"
        ↓
"What is uncertain?"
        ↓
"What information is missing?"
        ↓
"Can we prepare the application?"
        ↓
"Has the human approved it?"
        ↓
"Can it safely be submitted?"
ApplyX is designed around application decisioning, not just form filling.
👥 Built For
ApplyX is designed for people applying to:
Jobs
Internships
Scholarships
Other structured opportunities
The goal is to reduce repetitive work while improving the quality and transparency of application decisions.
🏆 Hackathon Focus
ApplyX demonstrates an agent that performs real work across an application workflow while maintaining deterministic controls and human authorization.
The project focuses on:
Agentic reasoning
Eligibility decisioning
Uncertainty handling
Clarification and re-evaluation
Application preparation
Human-in-the-loop authorization
Controlled browser automation
Traceable application state
📌 Core Differentiator
ApplyX is not just a form filler. It decides whether applying is worth doing — and keeps the human in control.
📄 License
This project is licensed under the MIT License.
See the LICENSE file for details.
🔐 Security Note
Never commit API keys, passwords, credentials, .env files, or other secrets to the repository.
For local development, keep secrets in environment variables or a local .env file that is excluded from Git.
💭 Final Thought
Don't just apply. Know why you're applying.
ApplyX — Autonomous in reasoning. Deterministic in control. Human in authorization.

