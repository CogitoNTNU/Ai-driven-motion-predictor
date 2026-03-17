"""LangGraph setup with supervisor and sub-agent for stock analysis."""

import os
from pathlib import Path
from typing import Annotated, List, Any, Dict
from typing_extensions import TypedDict, NotRequired
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor
from langchain.agents import create_agent
from langgraph.managed.is_last_step import RemainingStepsManager
from langgraph.graph import add_messages
from langchain_core.messages import AnyMessage
from agents.tools import get_stock_growth, get_current_price, get_stock_news_sentiment

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
    sub_agent_prompt = load_prompt("stock_analyst_sub_agent")
    main_agent_prompt = load_prompt("main_agent")
    sentiment_agent_prompt = load_prompt("sentiment_analyst_sub_agent")

    # Create the stock analyst sub-agent
    stock_analyst_sub_agent = create_agent(
        model=model,
        tools=[get_stock_growth, get_current_price],
        name="stock_analyst_sub_agent",
        system_prompt=sub_agent_prompt,
    )

    # Create the sentiment analyst sub-agent
    sentiment_analyst_sub_agent = create_agent(
        model=model,
        tools=[get_stock_news_sentiment],
        name="sentiment_analyst_sub_agent",
        system_prompt=sentiment_agent_prompt,
    )

    # Create the supervisor workflow with custom state
    workflow = create_supervisor(
        agents=[stock_analyst_sub_agent, sentiment_analyst_sub_agent],
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
