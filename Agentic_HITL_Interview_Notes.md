**Agentic HITL interview notes — Jay's KISS design**

**Memory key: Pause → Return → Review → Resume → Refresh.**

The current implementation uses a plain async section runner and independent question graph threads. The agreed minimal HITL line is:

```python
decision = interrupt("Awaiting human review")
```

The review UI obtains question details from saved QuestionState. A large interrupt payload is optional and unnecessary for this chosen UI flow.

**1. Architecture and the agentic part**

| Interview point | Precise answer |
|---|---|
| What are the three levels? | CAR contains section results. A section contains question results and ultimately its narrative. Each question owns its answer, evidence/citation IDs, guard results, recovery count and status. |
| Where is the agent? | In evidence recovery: the model diagnoses a gap and selects a focused query and allowlisted retrieval tool. Python executes the permitted tool and enforces the budget. |
| What does the normal question workflow do? | Retrieve candidates, merge/select evidence, draft an answer, audit citations, check claims and reasoning, then complete or recover. |
| What does evidence selection do? | Deduplicate chunk IDs using the highest score, sort by descending score with chunk-ID tie-breaking, and scan with a per-document cap. This function uses supplied scores; it does not itself perform semantic reranking. |
| Why three checks? | Citation audit checks ID membership; claim guard checks factual support; reasoning guard checks whether the answer and conclusions follow logically. Citation validity alone does not establish factual support. LLM guards are fallible. |
| What is bounded recovery? | At most two investigations after the initial attempt: up to three drafts. Merge new evidence and repeat selection, drafting and checks. Stop early when recovery supplies no new chunk IDs. |
| Is retrieval itself a tool? | Yes. Initial retrieval calls the search function with the original question. Recovery can reuse it, with the model choosing a focused query. |
| Is this a multi-agent system? | Multiple concurrent question runs are parallel workflow executions. Multiple nodes or LLM calls alone do not establish a multi-agent architecture. |

**2. What pause and resume actually mean**

| Interview point | Precise answer |
|---|---|
| What causes the pause? | Calling interrupt(). Setting status to awaiting_human is an application flag; it does not itself pause execution. |
| What happens to decision on the first call? | Nothing is assigned. The interrupt signals the runtime before the assignment completes. Code below that line does not run. |
| What returns to the outer caller? | The graph invocation returns its current output, with interrupt information under __interrupt__ in the default invocation API. Our runner reads checkpoint values instead. |
| What is the argument to interrupt()? | An outgoing notification/request. We use a short message because the UI can obtain the review details from saved state. |
| Where does the human see the question? | An application review screen supplied by the backend. It can use the collected QuestionState or fetch the latest question checkpoint. Viewing data does not execute human_review. |
| Where does Command(resume=...) come from? | Application code called after a human responds, outside the paused graph. It must target the original question thread. |
| Where does the decision dictionary come from? | The resume payload becomes interrupt()'s return value, and Python assigns it to decision. It is not inferred from the question or answer. |
| Does human_review start from its next line? | No. On resume, the node starts from its beginning. The interrupt call returns the supplied reply, allowing subsequent lines to execute. |
| Is human_review entered exactly twice? | Do not promise that. There is an initial entry and a resume entry; additional continuation attempts or retries can enter it again. Our section refresh may re-enter other still-paused questions. |
| Does approval rerun M05? | A normal resume at human_review uses the saved result of the separate, completed M05 node. It does not repeat M05. |
| What should precede interrupt()? | Prefer read-only work. Earlier code can repeat on node re-entry. Do not put an unprotected payment, notification or other side effect there. Do not swallow the interrupt signal in a broad exception handler. |

Official behavior: [LangGraph interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts) and the [interrupt/Command implementation](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/types.py).

For the already-built question_graph, an external approval of CAR 42, section 3, question 7 is:

```python
from langgraph.types import Command

await question_graph.ainvoke(
    Command(resume={
        "approved": True,
        "feedback": "Reviewed the evidence.",
    }),
    config={"configurable": {"thread_id": "42:3:7"}},
)
```

The minimal human-review node accepts that dictionary. Approval plus a nonempty answer gives completed; rejection gives failed. Automated guard results remain unchanged. Human approval is an explicit override, recorded in human_feedback.

**3. The 30-question batch**

| Interview point | Precise answer |
|---|---|
| Do we build 30 graph objects? | No. Reuse one compiled question graph with separate states and thread IDs. A LangGraph thread ID is a persistence identifier, not an operating-system thread. |
| What limits concurrent work? | The section's Semaphore(5). Thirty question calls are scheduled, with at most five inside the protected processing block. |
| What happens when question 7 pauses? | Its ainvoke returns. run_one reads its checkpoint state and returns that state. Exiting the semaphore block releases its slot. |
| If 25 finish and 5 need review? | When all 30 calls have returned, gather finishes. The returned SectionState contains 25 completed results and 5 awaiting_human results; section status is awaiting_human. |
| Can other sections continue? | Yes. The outer coordinator can collect this pending section and continue others. An awaiting_human section is not a completed section. |
| Where do question details remain? | section_state["question_results"][qid], plus the individual question checkpoint. Full question results preserve answer, IDs, checks and status. |
| Does approval automatically update the returned SectionState? | No. It updates the question checkpoint. Our approve_question helper then calls run_section_questions again to refresh the section. |
| Does that refresh redo completed work? | The runner returns saved values for finished threads. Still-paused questions remain awaiting review without an approval payload. |
| When is narrative generation allowed? | When every required question is completed. In our runner that makes the section running, ready for the narrative step; the section is not yet a finished narrative. |
| Is five a CAR-wide limit? | No. It is five per section. A CAR-wide limit would require sharing the semaphore across sections. |

**4. Checkpoints, state and idempotency**

| Interview point | Precise answer |
|---|---|
| What does a checkpoint retain? | Graph state and execution metadata, including pending work/interrupt information. It is not a saved Python stack or compiled graph object. |
| What is snapshot? | The StateSnapshot returned by aget_state(config), not a type we defined. values holds saved state; next identifies pending nodes. An empty next means execution finished, not necessarily business success. |
| Does reading a checkpoint execute the graph? | No. aget_state reads saved state. ainvoke executes or continues work. |
| Does ainvoke(None) approve a question? | No. It continues from saved execution without supplying a human answer. Command(resume=...) supplies that answer. |
| Where are our checkpoint boundaries? | M05 and human_review are separate nodes. The recovery loop inside M05 has no separate graph checkpoint per attempt. An unfinished M05 may rerun when execution is retried. |
| What happens on process restart? | InMemorySaver loses its data. Reuse the same graph/checkpointer instance in the demo. Restart recovery requires a durable backend and the same thread identity with compatible graph code. |
| What is our key? | car_id:section_id:question_id, supplied as configurable.thread_id. |
| Does the key guarantee idempotency? | No. It identifies saved work. Our runner also checks for an existing finished snapshot before invoking, avoiding repeated work on sequential calls. |
| Are simultaneous duplicate requests handled? | No locking is implemented. Production needs coordination/atomic deduplication for the same logical request. Changed inputs or report revisions also need an explicit key/version policy. |
| Why exclude raw chunk text? | We retain IDs and results. The question-node wrapper removes retrieved_chunks before returning, keeping raw text out of graph writes and nested question results. Evidence can be hydrated from IDs for review or narrative generation. |

Official behavior: [Checkpointers](https://docs.langchain.com/oss/python/langgraph/checkpointers) and [Persistence](https://docs.langchain.com/oss/python/langgraph/persistence).

**5. Follow-ups an interviewer may ask**

| Follow-up | Answer to give |
|---|---|
| What if the human never responds? | Saved status can remain awaiting_human while checkpoints are retained. No worker or semaphore slot is held. Expiry, reminders and escalation are application policies, not implemented here. |
| Does checkpointing prevent hung API calls? | No. LLM/search calls need their own timeouts and retry policy. Evidence recovery and technical retries solve different problems. |
| What if a search/model call raises an exception? | Our current section runner propagates technical exceptions. It does not yet convert each infrastructure failure into a per-question result. A HITL pause is a different, expected outcome. |
| What if the human edits the answer? | Our flow supports approving/rejecting the existing draft. An edit flow would accept the revision, update it and rerun relevant checks before accepting it. |
| How do you secure approvals in production? | Authenticate and authorize the reviewer, validate the reply, verify the pending question/revision, prevent conflicting duplicate submissions, and record reviewer identity, time and feedback. A thread ID is not authorization. |
| How do you debug a stuck question? | Locate its thread, inspect state status, next nodes, interrupt/error metadata and logs. Distinguish awaiting review from an API timeout or exception. |
| How does this apply to action-taking agents? | Interrupt before the action that requires approval. Resume with the decision, then execute only approved arguments and protect the side effect against duplicate execution. |

**6. Be precise about implementation scope**

- Implemented: evidence selection, question processing and guards, bounded recovery, the two-node question graph, question checkpoints, the async section runner, and an external approval helper that refreshes the section.
- Agreed simplification: use the short interrupt message above; the review screen reads QuestionState.
- Not built: the review UI/API endpoints, automatic approval notifications, section narrative generation and final CAR orchestration.
- The human-review node understands approved=False, but the current approve_question convenience helper sends approved=True. A Deny UI action still needs its caller wiring.
- The current section runner is ordinary async Python. Do not describe it as an already-implemented SectionGraph, or assume its independent-thread behavior automatically applies to every nested-graph design.
