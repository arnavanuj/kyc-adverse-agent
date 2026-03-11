from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.agents.compliance_guardrail_agent import ComplianceGuardrailAgent
from app.agents.planner_agent import PlannerAgent
from app.agents.reflection_agent import ReflectionAgent
from app.agents.report_generator_agent import ReportGeneratorAgent
from app.agents.risk_classification_agent import RiskClassificationAgent
from app.agents.scraper_agent import ScraperAgent
from app.agents.search_agent import SearchAgent
from app.agents.summarization_agent import SummarizationAgent
from app.db.memory import MemoryStore
from app.orchestrator.state import GraphState


class ScreeningWorkflow:
    def __init__(self, memory: MemoryStore):
        self.memory = memory

        self.planner = PlannerAgent()
        self.search = SearchAgent()
        self.scraper = ScraperAgent()
        self.risk = RiskClassificationAgent()
        self.summary = SummarizationAgent()
        self.reflection = ReflectionAgent()
        self.guardrail = ComplianceGuardrailAgent()
        self.reporter = ReportGeneratorAgent()

        self.graph = self._build_graph().compile()

    def _build_graph(self):
        workflow = StateGraph(GraphState)

        workflow.add_node("planner", self._planner_node)
        workflow.add_node("search", self._search_node)
        workflow.add_node("scrape", self._scrape_node)
        workflow.add_node("risk", self._risk_node)
        workflow.add_node("summarize", self._summarize_node)
        workflow.add_node("reflect", self._reflect_node)
        workflow.add_node("guardrail", self._guardrail_node)
        workflow.add_node("report", self._report_node)

        workflow.set_entry_point("planner")

        workflow.add_edge("planner", "search")
        workflow.add_edge("search", "scrape")
        workflow.add_edge("scrape", "risk")
        workflow.add_edge("risk", "summarize")
        workflow.add_edge("summarize", "reflect")

        workflow.add_conditional_edges(
            "reflect",
            self._reflection_router,
            {
                "search": "search",
                "guardrail": "guardrail",
            },
        )

        workflow.add_edge("guardrail", "report")
        workflow.add_edge("report", END)
        return workflow

    def _reflection_router(self, state: GraphState) -> str:
        if state.get("should_reflect_retry"):
            return "search"
        return "guardrail"

    async def _planner_node(self, state: GraphState) -> GraphState:
        out = await self.planner.run(state)
        await self._persist_messages(out)
        return out

    async def _search_node(self, state: GraphState) -> GraphState:
        out = await self.search.run(state)
        await self.memory.save_sources(out["case_id"], out.get("search_results", []))
        await self._persist_messages(out)
        return out

    async def _scrape_node(self, state: GraphState) -> GraphState:
        out = await self.scraper.run(state)
        await self.memory.save_sources(out["case_id"], out.get("articles", []))
        await self._persist_messages(out)
        return out

    async def _risk_node(self, state: GraphState) -> GraphState:
        out = await self.risk.run(state)
        await self.memory.save_findings(out["case_id"], out.get("findings", []))
        await self._persist_messages(out)
        return out

    async def _summarize_node(self, state: GraphState) -> GraphState:
        out = await self.summary.run(state)
        await self._persist_messages(out)
        return out

    async def _reflect_node(self, state: GraphState) -> GraphState:
        current = state.get("reflection_loop_count", 0)
        state["reflection_loop_count"] = current + 1
        out = await self.reflection.run(state)
        await self._persist_messages(out)
        return out

    async def _guardrail_node(self, state: GraphState) -> GraphState:
        out = await self.guardrail.run(state)
        await self._persist_messages(out)
        return out

    async def _report_node(self, state: GraphState) -> GraphState:
        out = await self.reporter.run(state)
        await self.memory.save_report(out["case_id"], out["report"])
        await self.memory.update_case_status(out["case_id"], out.get("status", "completed"))
        await self._persist_messages(out)
        return out

    async def _persist_messages(self, state: GraphState) -> None:
        messages = state.get("messages", [])
        if not messages:
            return

        while messages:
            msg = messages.pop(0)
            await self.memory.add_message(
                state["case_id"],
                msg.get("from", "unknown"),
                msg,
            )

    async def run(self, initial_state: GraphState) -> GraphState:
        return await self.graph.ainvoke(initial_state)
