import pandas as pd
from backtesting_opt import *
import slack_sdk
from slack_sdk.errors import SlackApiError

def send_slack_message(body):
    # Initialize the Slack client
    client = slack_sdk.WebClient(token="xoxb-4078811355153-6348770012323-hwup0fhCSlqZRUsiMtHoxExw")

    # Send a message to a channel
    try:
        response = client.chat_postMessage(
            channel="#pairs-bot",  # Replace with your desired channel name or ID
            text=body
        )
        print("Message sent successfully:", response["ts"])
    except SlackApiError as e:
        print("Error sending message:", e.response["error"])


df1 = getData(["NVDA", "AAPL"], start_date="2022-01-01",
              end_date="2023-09-01", timeframe='1h')
md = fit_ols(df1)

ls = latest_signal("NVDA","AAPL",'1h',14,14,md)
print(ls)

send_slack_message(f"{str(ls)}")
