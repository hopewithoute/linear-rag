import subprocess
import sys
import os
import time

datasets = ["hotpotqa", "musique", "medical"]

for ds in datasets:
    print(f"\n{'='*50}\nEVALUATING {ds.upper()}\n{'='*50}")
    
    env = os.environ.copy()
    if ds == "medical":
        env["NER_LABELS"] = "Disease,Symptom,Drug,Treatment,Protein,Pathogen,Gene,Virus,Bacteria,Medical Procedure"
    else:
        env["NER_LABELS"] = "Person,Location,Organization,Event,Date,Concept,Facility,Product,Creative Work,Law,Technology"
        
    print(f"Using NER_LABELS: {env['NER_LABELS']}")
    
    # Start GLiNER Sidecar using the current venv (which has CUDA)
    print("Starting GLiNER Sidecar...")
    sidecar_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8765"],
        cwd="../gliner-sidecar",
        env=os.environ.copy(),
        stdout=sys.stdout,
        stderr=sys.stderr
    )
    
    # Start Embedding Sidecar
    print("Starting Embedding Sidecar...")
    embedding_sidecar_process = subprocess.Popen(
        ["../embedding-sidecar/.venv/bin/python", "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8766"],
        cwd="../embedding-sidecar",
        env=os.environ.copy(),
        stdout=sys.stdout,
        stderr=sys.stderr
    )
    
    # Start API server in background
    print("Starting API Server...")
    api_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8003"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    # Wait until all servers are up
    import requests
    max_retries = 30
    for _ in range(max_retries):
        try:
            requests.get("http://127.0.0.1:8765/health", timeout=1)
            requests.get("http://127.0.0.1:8766/health", timeout=1)
            requests.get("http://127.0.0.1:8003/docs", timeout=1)
            print("All servers are ready!")
            break
        except requests.exceptions.RequestException:
            time.sleep(1)
    else:
        print("Servers failed to start within timeout.")
        api_process.terminate()
        sidecar_process.terminate()
        embedding_sidecar_process.terminate()
        continue
    
    try:
        # 1. Wipe DB
        print(f"Wiping database for {ds}...")
        subprocess.run([sys.executable, "scratch_reset.py"], check=True, env=env)
        
        # 2. Ingest
        print(f"Ingesting {ds}...")
        subprocess.run([sys.executable, "ingest_dataset.py", ds], check=True, env=env)
        
        # 3. Evaluate Retrieval
        print(f"Evaluating {ds}...")
        subprocess.run([sys.executable, "eval_retrieval_only.py", ds], check=True, env=env)
    finally:
        print("Stopping API Server...")
        api_process.terminate()
        api_process.wait()
        print("Stopping GLiNER Sidecar...")
        sidecar_process.terminate()
        sidecar_process.wait()
        print("Stopping Embedding Sidecar...")
        embedding_sidecar_process.terminate()
        embedding_sidecar_process.wait()
        
print("\nAll datasets evaluated!")
    
print("\nAll datasets evaluated!")
