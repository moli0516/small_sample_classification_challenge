import json
from pathlib import Path
from collections import Counter

import pandas as pd

diagnostic_files = sorted(
    Path("outputs/results").glob("*/e5_fold_diagnostics.csv")
)

for path in diagnostic_files:
    diagnostics = pd.read_csv(path)
    counts = Counter()

    for value in diagnostics["warnings_json"].dropna():
        for warning in json.loads(value):
            counts[
                (warning["category"], warning["message"])
            ] += 1

    print(f"\n{path}")

    for (category, message), count in counts.most_common():
        print(f"\n[{category}] {count} occurrences")
        print(message)