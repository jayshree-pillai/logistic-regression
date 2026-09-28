```python
import asyncio
import requests
from typing import Annotated, Literal
from typing_extensions import TypedDict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool, StructuredTool

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from langgraph.channels import UntrackedValue
```
```python
app = FastAPI()
class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)

    @field_validator("query", mode="before")
    @classmethod
    def normalize_query(cls, value):
        if not isinstance(value, str):
            raise ValueError("query must be a string")
        return " ".join(value.split())

class ChatResponse(BaseModel):
    answer: str = Field(strict=True, min_length=1, max_length=2000)

class ServiceUnavailableError(Exception):
    pass

async def generate_answer(query: str) -> str:
    return "Mock answer"

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        answer = await generate_answer(request.query)
    except TimeoutError as exc:
        raise HTTPException(504, detail="Answer generation timed out") from exc
    except ServiceUnavailableError as exc:
        raise HTTPException(503, detail="Answer service unavailable") from exc
    return ChatResponse(answer=answer)
```
```python
get_company_rating_tool = StructuredTool.from_function(
     func = AAA,name = "AAA", description=" AAA "
@tool
def get_company_sector(company_id: str) -> dict:
TOOLS = [get_company_rating_tool, get_company_sector]
llm = ChatOpenAI(model="gpt-5.6-sol",temperature=0)
llm_with_tools = llm.bind_tools(TOOLS)

def call_model(state: MessagesState):
    response = llm_with_tools.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
    )
    return {"messages": [response]}

builder = StateGraph(MessagesState)
builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode(TOOLS)) #Execute requested tools and return tool messages; note name must be tools for default
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition) #Route to `"tools"` when the model requests a tool; otherwise end. 
builder.add_edge("tools", "agent")
graph = builder.compile()
if __name__=="__main__":
    result = graph.invoke({"messages": [HumanMessage(content="What is ACME's credit Rating")]})
    for message in result["messages"]:
        message.pretty_print()
```

**M05 — one question, bounded recovery**
```python
MAX_RECOVERY_ATTEMPTS = 2
async def process_question(*,load_question,retrieve,generate_answer,llm,
    *,recovery_agent=recover_evidence,recovery_tools=None,) -> QuestionState:
    # 1. PREPARE — do this once.
    idempotency_key = f"{car_id}:{section_id}:{qid}"
    question = load_question(section_id, qid)
    tools = {"azure_search": retrieve} if recovery_tools is None else recovery_tools
    candidate_batches = [await retrieve(car_id, question)]
    state = QuestionState(...,idempotency_key=idempotency_key,  citation_audit_status="not_run", claim_guard_status="not_run",reasoning_guard_status="not_run",recovery_attempts=0,human_feedback=None,status="running",)
    for attempt in range(MAX_RECOVERY_ATTEMPTS + 1):
        chunks = select_evidence(candidate_batches, limit=8, max_per_document=2)
        # DRAFT an answer when evidence exists.
        if chunks:
            draft = await generate_answer(question, chunks)
            state["answer"] = draft["answer"]
            state["citation_chunk_ids"] = draft["citation_chunk_ids"]

        # CHECK citations first, then claims and reasoning.
        audit = audit_citations(state)
        state["citation_audit_status"] = "passed" if audit["passed"] else "failed"
        if audit["passed"]:
            claims = await claim_guard(state, llm)
            reasoning = await reasoning_guard(state, llm)
            state["claim_guard_status"] = claims["claim_guard_status"]
            state["reasoning_guard_status"] = reasoning["reasoning_guard_status"]
        # DECISION 1: passed? Finish immediately.
        passed = (
            state["citation_audit_status"] == "passed"
            and state["claim_guard_status"] == "passed"
            and state["reasoning_guard_status"] == "passed"
        )
        if passed:
            state["status"] = "completed"
            return state

        # DECISION 2: no recovery budget left? Stop the loop.
        if attempt == MAX_RECOVERY_ATTEMPTS:
            break

        # RECOVER once. The helper returns another list of chunks.
        state["recovery_attempts"] += 1
        new_chunks = await recovery_agent(
            state=state, car_id=car_id, llm=llm, tools=tools
        )

        # DECISION 3: no new evidence? Stop the loop.
        seen_ids = {c["chunk_id"] for batch in candidate_batches for c in batch}
        if not any(c["chunk_id"] not in seen_ids for c in new_chunks):
            break

        # New evidence will be selected and checked on the next loop pass.
        candidate_batches.append(new_chunks)

    # 3. FALLBACK — reached after either break above.
    state["status"] = "awaiting_human"
    return state            
```
**M05_question_graph.py — wrap → route → checkpoint**
```python
from M0_CAR_Defns import QuestionState
from M05_bounded_evidence_recovery import process_question
from M05_hitl import human_review

def build_question_graph(load_question, retrieve, generate_answer, llm):
    builder = StateGraph(QuestionState)

    async def question_node(state: QuestionState) -> dict:
        result = await process_question(
            car_id=state["car_id"],
            section_id=state["section_id"],
            qid=state["question_id"],
            load_question=load_question,
            retrieve=retrieve,
            generate_answer=generate_answer,
            llm=llm,
        )
        result.pop("retrieved_chunks", None)
        return result

    def route_after_question(state: QuestionState):
        return "human" if state["status"] == "awaiting_human" else "done"

    builder.add_node("question_node", question_node)
    builder.add_node("human_review", human_review)

    builder.add_edge(START, "question_node")
    builder.add_conditional_edges(
        "question_node",
        route_after_question,
        {"human": "human_review", "done": END},
    )
    builder.add_edge("human_review", END)

    return builder.compile(checkpointer=InMemorySaver())
```
**M05_hitl.py — interrupt → receive decision → return status**
```python
from langgraph.types import interrupt
from M0_CAR_Defns import QuestionState
def human_review(state: QuestionState) -> dict:
    decision = interrupt("Awaiting human review")

    approved = decision["approved"] is True and bool(state["answer"])
    feedback = decision.get("feedback", "")

    return {
        "status": "completed" if approved else "failed",
        "human_feedback": (
            f"{'Approved' if approved else 'Rejected'}: {feedback}"
        ),
    }
```
**M05_resume_question.py — same graph, same thread, human decision**
```python
from langgraph.types import Command
async def resume_question(
    question_graph,
    car_id: int,
    section_id: int,
    qid: int,
    approved: bool,
    feedback: str = "",
) -> dict:
    key = f"{car_id}:{section_id}:{qid}"
    config = {"configurable": {"thread_id": key}}
    await question_graph.ainvoke(
        Command(resume={"approved": approved, "feedback": feedback}),
        config=config,
    )
    return (await question_graph.aget_state(config)).values
```
