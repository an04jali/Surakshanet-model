import os
import time
import requests

# Telegram configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Prevent repeated alerts
last_alert_time = 0
ALERT_COOLDOWN = 10


def send_telegram_alert(risk, count, movement=0):
    """
    Sends a Telegram alert when suspicious/high-risk activity is detected.
    """

    global last_alert_time

    # Don't send if credentials are missing
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram credentials not configured.")
        return False

    # Prevent alert spam
    current_time = time.time()

    if current_time - last_alert_time < ALERT_COOLDOWN:
        return False

    message = (
        "🚨 SURAKSHANET ALERT 🚨\n\n"
        f"Risk Level: {risk}\n"
        f"People Detected: {count}\n"
        f"Movement Score: {movement}\n\n"
        "Suspicious activity detected."
    )

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    try:
        response = requests.post(
            url,
            data={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message
            },
            timeout=5
        )

        if response.ok:
            last_alert_time = current_time
            print("Telegram alert sent successfully.")
            return True

        print("Telegram error:", response.text)
        return False

    except Exception as e:
        print("Telegram connection error:", e)
        return False


def trigger_alert(risk, count, movement=0):
    """
    Main SurakshaNet alert function.
    """

    if risk == "High" and count > 10:
        send_telegram_alert(risk, count, movement)