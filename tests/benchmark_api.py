
import time
import statistics
import requests

API_URL = "http://127.0.0.1:7860"

for batch_size in [1, 10, 50, 100]:
    payload = {
        "comments": ["This video is amazing!"] * batch_size
    }

    times = []

    for _ in range(5):
        start = time.perf_counter()

        response = requests.post(
            f"{API_URL}/predict_batch",
            json=payload,
            timeout=60
        )
        response.raise_for_status()

        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)

    print(
        f"Batch {batch_size:3d} | "
        f"Latence moyenne : {statistics.mean(times):.2f} ms | "
        f"Min : {min(times):.2f} ms | "
        f"Max : {max(times):.2f} ms"
    )
