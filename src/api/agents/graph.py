"""LangGraph setup with supervisor and sub-agent for stock analysis."""

import os
from pathlib import Path
from typing import Annotated, List, Any, Dict
from typing_extensions import TypedDict, NotRequired
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from langgraph.managed.is_last_step import RemainingStepsManager
from langgraph.graph import add_messages
from langchain_core.messages import AnyMessage
from agents.tools import get_stock_growth, get_current_price

# Load environment variables
load_dotenv()


class ChartData(TypedDict):
    """Schema for chart artifacts."""

    chart_id: str
    symbol: str
    type: str
    data: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class AgentState(TypedDict):
    """Custom state schema with chart support."""

    messages: Annotated[list[AnyMessage], add_messages]
    charts: List[ChartData]
    remaining_steps: NotRequired[Annotated[int, RemainingStepsManager]]


def load_prompt(prompt_name: str) -> str:
    """Load a prompt from the prompts directory."""
    prompt_path = Path(__file__).parent.parent / "prompts" / f"{prompt_name}.md"
    with open(prompt_path, "r") as f:
        return f.read()


def extract_chart_artifacts(tool_results: List[Any]) -> List[ChartData]:
    """Extract chart artifacts from tool results."""
    charts = []
    for result in tool_results:
        if isinstance(result, dict) and result.get("chart_artifact"):
            charts.append(result["chart_artifact"])
    return charts


def create_stock_agents():
    """Create and configure the supervisor and sub-agent workflow."""

    # Initialize the LLM with custom endpoint
    model = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "openai/gpt-oss-120b"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.1,
        streaming=True,
    )

    # Load prompts
    sub_agent_prompt = load_prompt("sub_agent")
    main_agent_prompt = load_prompt("main_agent")

    # Create the sub-agent (stock analyst)
    stock_analyst = create_react_agent(
        model=model,
        tools=[get_stock_growth, get_current_price],
        name="stock_analyst",
        prompt=sub_agent_prompt,
    )

    # Create the supervisor workflow with custom state
    workflow = create_supervisor(
        agents=[stock_analyst],
        model=model,
        prompt=main_agent_prompt,
        state_schema=AgentState,
    )

    # Compile the graph
    app = workflow.compile()

    return app


# Global instance
graph_app = None


def get_graph():
    """Get or create the graph application."""
    global graph_app
    if graph_app is None:
        graph_app = create_stock_agents()
    return graph_app
