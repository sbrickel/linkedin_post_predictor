#!/usr/bin/env python3
"""
Semi-automated helper script to collect LinkedIn post text manually.

Opens each post URL in your browser, waits for your input (multi-line paste supported),
and saves the results to data/post_texts.csv.

End input with a line containing only "::done".
"""

import pandas as pd
import webbrowser
import time
import os
from pathlib import Path

# --- Configuration ---
INPUT_CSV = "data/linkedin_posts_clean.csv"
OUTPUT_CSV = "data/post_texts.csv"
DELAY_BETWEEN = 5  # seconds before prompting (time to load page)


def main():
    print("📂 Loading LinkedIn posts from:", INPUT_CSV)
    df = pd.read_csv(INPUT_CSV)

    if "post_url" not in df.columns:
        raise ValueError("❌ No 'post_url' column found in the input CSV!")

    # Load existing progress if available
    if os.path.exists(OUTPUT_CSV):
        done_df = pd.read_csv(OUTPUT_CSV)
        done_urls = set(done_df["post_url"])
        print(f"🔁 Resuming: {len(done_urls)} posts already collected.")
    else:
        done_df = pd.DataFrame(columns=["post_url", "post_text"])
        done_urls = set()
        print("🆕 Starting new text collection session.")

    Path("data").mkdir(exist_ok=True)

    remaining = df[~df["post_url"].isin(done_urls)]
    print(f"🚀 Ready to process {len(remaining)} remaining posts.\n")

    for i, row in remaining.iterrows():
        url = row["post_url"]

        if pd.isna(url):
            continue

        print(f"\n🔗 Opening post {i+1}/{len(df)}:\n{url}")
        webbrowser.open_new_tab(url)
        time.sleep(DELAY_BETWEEN)

        print("📝 Paste the post text below. Type '::done' on a new line when finished:")

        # Collect multiline input until ::done
        lines = []
        while True:
            line = input()
            if line.strip() == "::done":
                break
            lines.append(line)

        post_text = "\n".join(lines).strip()

        if not post_text:
            print("⚠️ No text entered — skipping this post.")
            continue

        # Save incrementally
        new_entry = pd.DataFrame([{"post_url": url, "post_text": post_text}])
        done_df = pd.concat([done_df, new_entry], ignore_index=True)
        done_df.to_csv(OUTPUT_CSV, index=False)

        print(f"✅ Saved text for this post ({len(post_text.split())} words).")

    print("\n🎉 All done! Collected texts saved to:", OUTPUT_CSV)


if __name__ == "__main__":
    main()

