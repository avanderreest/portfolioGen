import autogen
import pandas as pd
import requests
from typing import List, Dict

# Define the configuration for the agents
config_list = autogen.config_list_from_json("OAI_CONFIG_LIST")

# Create the assistant agent
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config={
        "config_list": config_list,
    },
)

# Create a custom user proxy agent to handle the specific tasks
class PortfolioAnalysisAgent(autogen.UserProxyAgent):
    def __init__(self, name, human_input_mode="NEVER", **kwargs):
        super().__init__(name, human_input_mode, **kwargs)

    def run_python_script(self):
        import subprocess
        subprocess.run(["python", "generate_portfolio.py"])

    def read_watchlist(self) -> List[str]:
        df = pd.read_csv("watchlist.csv")
        return df["Ticker"].tolist()

    def query_perplexity(self, ticker: str) -> str:
        api_key = "YOUR_PERPLEXITY_API_KEY"
        headers = {"Authorization": f"Bearer {api_key}"}
        query = f"Estimate the growth potential for the stock ticker {ticker}"
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json={
                "model": "mixtral-8x7b-instruct",
                "messages": [{"role": "user", "content": query}]
            }
        )
        return response.json()["choices"][0]["message"]["content"]

    def analyze_growth_potential(self, tickers: List[str]) -> List[Dict[str, str]]:
        results = []
        for ticker in tickers:
            growth_potential = self.query_perplexity(ticker)
            results.append({"ticker": ticker, "growth_potential": growth_potential})
        return sorted(results, key=lambda x: x["growth_potential"], reverse=True)

    def initiate_chat(self):
        self.run_python_script()
        tickers = self.read_watchlist()
        ranked_tickers = self.analyze_growth_potential(tickers)
        
        report = "Ranked tickers based on growth potential:\n"
        for item in ranked_tickers:
            report += f"{item['ticker']}: {item['growth_potential']}\n"
        
        self.send(report, assistant)

# Create the portfolio analysis agent
portfolio_agent = PortfolioAnalysisAgent(
    name="portfolio_analyst",
    human_input_mode="NEVER",
    llm_config={
        "config_list": config_list,
    },
)

# Start the conversation
portfolio_agent.initiate_chat()