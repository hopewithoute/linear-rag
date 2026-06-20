import sys
sys.path.append("../gliner-sidecar")
from gliner import GLiNER
import time

model = GLiNER.from_pretrained("/var/www/grasp-next/.models/gliner_multi-v2.1").to("cuda")

texts = ["Hello, I am John Doe from OpenAI.", "This is a test sentence in Jakarta, Indonesia."] * 50

print(f"Testing {len(texts)} sentences...")

t0 = time.time()
try:
    res = model.batch_predict_entities(texts, ["Person", "Location", "Organization"], batch_size=8)
    print(f"batch_predict_entities took {time.time()-t0:.2f}s")
except Exception as e:
    print(f"batch_predict_entities failed: {e}")

t0 = time.time()
try:
    res = []
    for t in texts:
        res.append(model.predict_entities(t, ["Person", "Location", "Organization"]))
    print(f"sequential predict_entities took {time.time()-t0:.2f}s")
except Exception as e:
    print(f"sequential predict_entities failed: {e}")
