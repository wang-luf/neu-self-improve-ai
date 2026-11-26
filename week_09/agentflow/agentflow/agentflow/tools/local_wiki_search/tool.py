"""Local Wikipedia search tool (assignment addition).

Replaces Google_Search_Tool (Gemini API) and Wikipedia_Search_Tool (OpenAI API)
with dense retrieval over the wiki-18 corpus (Search-R1's E5 Flat index),
served by `week_09/retriever/gpu_retrieval_server.py`. No API key is needed.
"""

import os

import requests

from agentflow.tools.base import BaseTool

TOOL_NAME = "Local_Wikipedia_Search_Tool"

LIMITATIONS = """
1. Searches a fixed 2018 English Wikipedia snapshot, not the live web.
2. Returns the top passages for the query; it does not summarize them.
3. Each passage is a 100-word chunk, so one search may not contain the full answer.
"""

BEST_PRACTICES = """
1. Use this tool to look up facts about people, places, organizations, events, and dates.
2. Search for one entity or fact at a time; for multi-hop questions, search again with the entity found in the previous step.
3. Use short, specific queries such as "Citibank founding year" rather than full questions.
"""


class Local_Wiki_Search_Tool(BaseTool):
    def __init__(self, model_string=None):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description="A Wikipedia search tool that returns the most relevant Wikipedia passages for a query.",
            tool_version="1.0.0",
            input_types={"query": "str - The search query, e.g. an entity name and the fact to find."},
            output_type="str - The top Wikipedia passages, each with its article title.",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(query="Citibank founding year")',
                    "description": "Find the year Citibank was founded.",
                },
                {
                    "command": 'execution = tool.execute(query="President of the United States in 1812")',
                    "description": "Find who was U.S. president in a given year.",
                },
            ],
            user_metadata={"limitations": LIMITATIONS, "best_practices": BEST_PRACTICES},
        )
        self.url = os.environ.get("WIKI_RETRIEVER_URL", "http://127.0.0.1:8002/retrieve")
        self.topk = int(os.environ.get("WIKI_RETRIEVER_TOPK", "3"))

    def execute(self, query: str):
        response = requests.post(self.url, json={"queries": [query], "topk": self.topk}, timeout=60)
        response.raise_for_status()
        passages = []
        for i, doc in enumerate(response.json()["result"][0], start=1):
            title, _, text = doc["contents"].partition("\n")
            passages.append(f"Doc {i} (Title: {title.strip(chr(34))}) {text.strip()}")
        return "\n".join(passages) or "No relevant Wikipedia passages were found."

    def get_metadata(self):
        return super().get_metadata()


if __name__ == "__main__":
    print(Local_Wiki_Search_Tool().execute(query="Citibank founding year"))
