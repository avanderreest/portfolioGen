# get_alpaca_positions.py

import os
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
import pandas as pd
from pathlib import Path

def get_positions():
    """
    Retrieves the list of tickers for which the account holds open positions.
    Returns a list of ticker symbols.
    """
    # Load API credentials from .env file in the parent directory
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(dotenv_path=env_path)
    API_KEY = os.getenv('API_KEY')
    API_SECRET = os.getenv('SECRET_KEY')
    BASE_URL = 'https://paper-api.alpaca.markets'  # Use 'https://api.alpaca.markets' for live trading

    if not API_KEY or not API_SECRET:
        print("API_KEY and API_SECRET must be set in the .env file.")
        return None

    # Initialize the Alpaca API
    api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

    try:
        # Get all open positions
        positions = api.list_positions()

        if not positions:
            print("No open positions found.")
            return []

        # Extract tickers from positions
        tickers = [position.symbol for position in positions]
        return tickers

    except Exception as e:
        print(f"An error occurred while fetching positions: {e}")
        return None

def read_watchlist():
    """
    Reads 'watchlist.csv' and extracts tickers from the 'Symbol' column.
    Returns a list of ticker symbols.
    """
    try:
        df = pd.read_csv('watchlist.csv')
        tickers = df['Symbol'].tolist()
        print("Extracted ticker list from watchlist.csv:", tickers)
        return tickers
    except Exception as e:
        print(f"An error occurred while reading watchlist.csv: {e}")
        return []

def main():
    positions_tickers = get_positions()
    if positions_tickers is None:
        print("Failed to retrieve positions.")
    else:
        # Read watchlist tickers
        watchlist_tickers = read_watchlist()

        # Create lists with origin for each ticker
        positions_list = [{'Ticker': ticker, 'Origin': 'position'} for ticker in positions_tickers]
        watchlist_list = [{'Ticker': ticker, 'Origin': 'watchlist'} for ticker in watchlist_tickers]

        # Merge the two lists
        combined_list = positions_list + watchlist_list
        
        # Find new alerts that need to be added
        new_alerts = [item for item in watchlist_list if item not in positions_list]

        # Save new alerts to a file called new_alerts.csv
        new_alerts_df = pd.DataFrame(new_alerts)
        new_alerts_df.to_csv('new_tobe_added_alerts.csv', index=False)
                

        # Create a DataFrame
        combined_df = pd.DataFrame(combined_list)

        # Assign a priority to 'Origin' to ensure 'position' comes before 'watchlist'
        combined_df['OriginPriority'] = combined_df['Origin'].map({'position': 0, 'watchlist': 1})

        # Sort by 'Ticker' and 'OriginPriority'
        combined_df.sort_values(by=['Ticker', 'OriginPriority'], inplace=True)

        # Drop duplicates on 'Ticker', keeping the first occurrence (which will be 'position' if there is a duplicate)
        combined_df.drop_duplicates(subset='Ticker', keep='first', inplace=True)

        # Drop the 'OriginPriority' column as it's no longer needed
        combined_df.drop(columns='OriginPriority', inplace=True)

        # Reset index if desired (optional)
        combined_df.reset_index(drop=True, inplace=True)

        # Print the combined DataFrame
        print("Combined Ticker List with Unique Tickers:")
        print(combined_df)

        # Optionally, save the combined list to a CSV file without headers
        combined_df[['Ticker']].to_csv('tradingview_watchlist.csv', index=False, header=False)

if __name__ == "__main__":
    main()