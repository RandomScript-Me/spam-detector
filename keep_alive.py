import threading
import requests
import time

def ping():
    while True:
        try:
            requests.get("https://spam-detector-g27s.onrender.com")
            print("Pinged server to keep alive")
        except:
            pass
        time.sleep(600)  # ping every 10 minutes

def start():
    t = threading.Thread(target=ping)
    t.daemon = True
    t.start()