import os
import requests
from datetime import datetime

def send_discord(message: str, color: int = 3066993, title: str = "ML Pipeline Notification"):
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        print("DISCORD_WEBHOOK_URL not set, skipping Discord notification.")
        return

    data = {
        "embeds": [
            {
                "title": title,
                "description": message,
                "color": color,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        ]
    }
    
    try:
        response = requests.post(webhook_url, json=data)
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to send Discord notification: {e}")
