"""A stand-in for a product's own scorer, speaking the stdin/stdout protocol.

stdin {"case": {...}, "output_text": "..."}; stdout {"metrics": {...}, "flags": [...]}.
Words in the output steer it, so a test can ask for each failure mode:
SCORER_CRASH exits 3, SCORER_GARBAGE prints non-JSON, SCORER_SLEEP sleeps past any timeout.
"""

import json
import sys
import time

payload = json.load(sys.stdin)
text = payload["output_text"]
if "SCORER_CRASH" in text:
    print("boom", file=sys.stderr)
    sys.exit(3)
if "SCORER_GARBAGE" in text:
    print("not json")
    sys.exit(0)
if "SCORER_SLEEP" in text:
    time.sleep(30)
label = payload["case"]["grading_context"].get("labels", {}).get("type", "")
words = len(text.split())
flags = ["too_short"] if words < 20 else []
print(
    json.dumps(
        {
            "metrics": {
                "word_count": words,
                "names_type": label.replace("_", " ") in text.lower(),
            },
            "flags": flags,
        }
    )
)
