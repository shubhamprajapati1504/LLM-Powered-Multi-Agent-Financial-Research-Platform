# A sample list of NIFTY 100 stocks (Name, Ticker)
# You can expand this list significantly for better coverage
NIFTY_100_STOCKS = [
    ("Adani Enterprises Ltd.", "ADANIENT.NS"),
    ("Adani Green Energy Ltd.", "ADANIGREEN.NS"),
    ("Adani Ports and Special Economic Zone Ltd.", "ADANIPORTS.NS"),
    ("Adani Total Gas Ltd.", "ATGL.NS"), # Example, check actual inclusion
    ("Apollo Hospitals Enterprise Ltd.", "APOLLOHOSP.NS"),
    ("Asian Paints Ltd.", "ASIANPAINT.NS"),
    ("Axis Bank Ltd.", "AXISBANK.NS"),
    ("Bajaj Auto Ltd.", "BAJAJ-AUTO.NS"),
    ("Bajaj Finance Ltd.", "BAJFINANCE.NS"),
    ("Bajaj Finserv Ltd.", "BAJAJFINSV.NS"),
    ("Bharti Airtel Ltd.", "BHARTIARTL.NS"),
    ("Bharat Petroleum Corporation Ltd.", "BPCL.NS"),
    ("Britannia Industries Ltd.", "BRITANNIA.NS"),
    ("Cipla Ltd.", "CIPLA.NS"),
    ("Coal India Ltd.", "COALINDIA.NS"),
    ("Divi's Laboratories Ltd.", "DIVISLAB.NS"),
    ("Dr. Reddy's Laboratories Ltd.", "DRREDDY.NS"),
    ("Eicher Motors Ltd.", "EICHERMOT.NS"),
    ("Grasim Industries Ltd.", "GRASIM.NS"),
    ("HCL Technologies Ltd.", "HCLTECH.NS"),
    ("HDFC Bank Ltd.", "HDFCBANK.NS"),
    ("HDFC Life Insurance Company Ltd.", "HDFCLIFE.NS"),
    ("Hero MotoCorp Ltd.", "HEROMOTOCO.NS"),
    ("Hindalco Industries Ltd.", "HINDALCO.NS"),
    ("Hindustan Unilever Ltd.", "HINDUNILVR.NS"),
    ("ICICI Bank Ltd.", "ICICIBANK.NS"),
    ("IndusInd Bank Ltd.", "INDUSINDBK.NS"),
    ("Infosys Ltd.", "INFY.NS"),
    ("ITC Ltd.", "ITC.NS"),
    ("JSW Steel Ltd.", "JSWSTEEL.NS"),
    ("Kotak Mahindra Bank Ltd.", "KOTAKBANK.NS"),
    ("Larsen & Toubro Ltd.", "LT.NS"),
    ("Mahindra & Mahindra Ltd.", "M&M.NS"),
    ("Maruti Suzuki India Ltd.", "MARUTI.NS"),
    ("Nestle India Ltd.", "NESTLEIND.NS"),
    ("NTPC Ltd.", "NTPC.NS"),
    ("Oil & Natural Gas Corporation Ltd.", "ONGC.NS"),
    ("Power Grid Corporation of India Ltd.", "POWERGRID.NS"),
    ("Reliance Industries Ltd.", "RELIANCE.NS"),
    ("State Bank of India", "SBIN.NS"),
    ("Sun Pharmaceutical Industries Ltd.", "SUNPHARMA.NS"),
    ("Tata Consultancy Services Ltd.", "TCS.NS"),
    ("Tata Consumer Products Ltd.", "TATACONSUM.NS"),
    ("Tata Motors Ltd.", "TATAMOTORS.NS"),
    ("Tata Steel Ltd.", "TATASTEEL.NS"),
    ("Tech Mahindra Ltd.", "TECHM.NS"),
    ("Titan Company Ltd.", "TITAN.NS"),
    ("UltraTech Cement Ltd.", "ULTRACEMCO.NS"),
    ("Wipro Ltd.", "WIPRO.NS"),
    # Add more stocks...
]

# Create mappings for easier lookup
NAME_TO_TICKER = {name.lower(): ticker for name, ticker in NIFTY_100_STOCKS}
TICKER_TO_NAME = {ticker: name for name, ticker in NIFTY_100_STOCKS}
STOCK_NAMES = [name.lower() for name, ticker in NIFTY_100_STOCKS]


# Benchmark Index
BENCHMARK_INDEX = "^NSEI" # NIFTY 50