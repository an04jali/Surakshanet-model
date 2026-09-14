import time

last_alert_time = 0

def trigger_alert(risk, count):
    global last_alert_time

    if risk == "High" and count > 10:
        if time.time() - last_alert_time > 3:
            print("⚠️ HIGH RISK ALERT - Crowd count:", count)
            last_alert_time = time.time()