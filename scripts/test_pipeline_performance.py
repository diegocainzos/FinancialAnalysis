import time
import subprocess
import sqlite3
import json

def run_command(cmd, desc):
    print(f"\n--- Running: {desc} ---")
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end = time.time()
    
    elapsed = end - start
    print(f"Time taken: {elapsed:.2f} seconds")
    if result.returncode != 0:
        print(f"Error output:\n{result.stderr}")
    else:
        print(f"Output:\n{result.stdout.strip()}")
    return elapsed

def main():
    print("Testing pipeline speed and functionality...")
    
    # 1. Time ingestion
    # Let's limit it to 20 per query to not explode entirely but get a decent chunk
    ingest_cmd = [
        ".venv/bin/python", "-m", "pipeline.ingest_bluesky", 
        "--limit", "20", "--concurrency", "5"
    ]
    ingest_time = run_command(ingest_cmd, "Ingesting Bluesky posts")
    
    # 2. Time processing
    process_cmd = [
        ".venv/bin/python", "-m", "pipeline.process_sentiment", 
        "--limit", "100", "--model", "finbert", 
        "--relevance-filter", "llm"
    ]
    process_time = run_command(process_cmd, "Processing Sentiment & LLM relevance")
    
    # 3. Check results in DB
    print("\n--- Verifying Relevance of Results ---")
    db_path = "data/sentiment.db"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    cur.execute("""
        SELECT rd.text, sr.sentiment_label, sr.sentiment_score
        FROM sentiment_results sr
        JOIN raw_documents rd ON sr.raw_document_id = rd.id
        ORDER BY sr.processed_at DESC
        LIMIT 5
    """)
    rows = cur.fetchall()
    print("\nLatest Processed Documents:")
    for row in rows:
        text, label, score = row
        print(f"Label: {label:<8} | Score: {score:.3f}")
        print(f"Text: {text}")
        print("-" * 50)
        
    conn.close()

if __name__ == "__main__":
    main()
