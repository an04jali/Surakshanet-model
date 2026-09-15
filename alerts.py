import os
import time
import requests

# Telegram configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Prevent repeated alerts
last_alert_time = 0
ALERT_COOLDOWN = 10


def send_telegram_alert(risk, count, movement=0, source="Browser Webcam"):
    """
    Sends a Telegram alert when suspicious/high-risk activity is detected.
    Returns True if Telegram message was successfully sent.
    """

    global last_alert_time

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram credentials not configured.")
        return False

    current_time = time.time()

    # Prevent alert spam
    if current_time - last_alert_time < ALERT_COOLDOWN:
        return False

    message = (
        "🚨 SURAKSHANET ALERT 🚨\n\n"
        f"Source: {source}\n"
        f"Risk Level: {risk}\n"
        f"People Detected: {count}\n"
        f"Movement Score: {movement}\n\n"
        "Suspicious activity detected."
    )

    url = (
        f"https://api.telegram.org/"
        f"bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    )

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


def trigger_alert(
    risk,
    count,
    movement=0,
    source="Browser Webcam"
):
    """
    Main SurakshaNet alert function.
    Returns True only when a Telegram alert is actually sent.
    """

    if risk == "High" and count > 10:
        return send_telegram_alert(
            risk,
            count,
            movement,
            source
        )

    return False