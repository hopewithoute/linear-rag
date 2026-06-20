import os
import subprocess
import time
import sys
import requests

def wait_for_server(url, timeout=30):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url)
            if r.status_code == 200:
                return True
        except:
            pass
        time.sleep(1)
    return False

def main():
    env = os.environ.copy()
    
    print("Starting API Server...")
    api_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8765"],
        cwd=".",
        env=env
    )
    
    try:
        if not wait_for_server("http://127.0.0.1:8765/health"): raise Exception("API failed")
        
        print("All servers are ready! Running SQuAD eval...")
        subprocess.run([sys.executable, "eval_squad.py"], check=True, env=env)
        
    finally:
        print("Stopping servers...")
        api_process.terminate()
        api_process.wait()

if __name__ == "__main__":
    main()
