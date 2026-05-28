import requests
import time

url = "https://retailpulse-ai-customer-analytics-and-demand-forecasting-z9lmk.streamlit.app/"

success = 0
failed = 0

start = time.time()

for i in range(20):

    response = requests.get(url)

    if response.status_code == 200:
        success += 1
    else:
        failed += 1

    print(f"Request {i+1}: {response.status_code}")

end = time.time()

print("Successful:", success)
print("Failed:", failed)
print("Execution Time:", round(end-start,2))