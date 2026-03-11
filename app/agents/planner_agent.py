from __future__ import annotations

from app.orchestrator.state import GraphState


class PlannerAgent:
    name = "planner_agent"

    async def run(self, state: GraphState) -> GraphState:
        full_name = state["full_name"]
        country = state.get("country")

        queries = [
            f'"{full_name}" adverse media',
            f'"{full_name}" fraud OR laundering OR sanctions',
            f'"{full_name}" indictment OR convicted OR bribery',
        ]

        if country:
            queries.append(f'"{full_name}" {country} investigation')

        plan = {
            "search_depth": len(queries),
            "need_reflection": True,
            "max_reflection_loops": 1,
            "query_strategy": "name + risk-keyword expansion",
        }

        state["queries"] = queries
        state["plan"] = plan
        state.setdefault("messages", []).append(
            {
                "from": self.name,
                "to": "search_agent",
                "type": "plan",
                "payload": {"queries": queries, "plan": plan},
            }
        )
        return state
