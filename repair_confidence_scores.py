"""
repair_confidence_scores.py
----------------------------
One-time repair script for DF-05: a handful of predictions made right
after switching to the XGBoost model had their confidence_score stored
as a raw 4-byte BLOB instead of a number (see db._to_native for why).

This script finds any row where confidence_score is stored as bytes,
decodes it back to the original float32 value it actually was, and
rewrites it as a proper REAL number. It is safe to run multiple times
-- rows that are already numeric are left untouched.

Run once, after updating db.py and app.py, and before collecting any
further evidence for TC-08:

    python repair_confidence_scores.py
"""

import sqlite3
import struct

DB_PATH = "cvd_dashboard.db"


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT id, confidence_score FROM predictions").fetchall()

    repaired = 0
    for row in rows:
        value = row["confidence_score"]
        if isinstance(value, bytes):
            try:
                (decoded,) = struct.unpack("<f", value)
            except struct.error:
                print(f"  Skipping id={row['id']}: could not decode {value!r}")
                continue
            decoded = round(decoded, 2)
            conn.execute(
                "UPDATE predictions SET confidence_score = ? WHERE id = ?",
                (decoded, row["id"]),
            )
            print(f"  Repaired id={row['id']}: {value!r}  ->  {decoded}")
            repaired += 1

    conn.commit()
    conn.close()

    if repaired:
        print(f"\nDone. Repaired {repaired} row(s).")
    else:
        print("No corrupted rows found -- nothing to repair.")


if __name__ == "__main__":
    main()
