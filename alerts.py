import winsound
import time

last_alert_time = 0

def trigger_alert(risk, count):
    global last_alert_time

    if risk == "High" and count > 10:
        if time.time() - last_alert_time > 3:
            winsound.Beep(2000, 500)
            last_alert_time = time.time()