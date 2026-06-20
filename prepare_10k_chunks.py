import json
import random
import sys
import os

try:
    from datasets import load_dataset
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
    from datasets import load_dataset

def main():
    print("Loading 2WikiMultihopQA validation dataset from raw JSON...")
    dev_path = '/home/hopewithoute/.cache/huggingface/hub/datasets--voidful--2WikiMultihopQA/snapshots/16852fde9d85cba158cf7e6517e7a3f9415a28c0/dev.json'
    
    with open(dev_path, 'r') as f:
        # Some HuggingFace datasets are JSON Lines, some are JSON arrays. 
        # We can read line by line if JSON Lines, or parse whole if array.
        try:
            val_data = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            val_data = [json.loads(line) for line in f]
    print(f"Loaded {len(val_data)} validation samples.")
    
    # Load our 1000 questions
    q_path = '/var/www/grasp-next/services/linear-rag/dataset/2wikimultihop/questions.json'
    with open(q_path, 'r') as f:
        questions = json.load(f)
        
    q_ids = set(q['id'] for q in questions)
    
    # Gather chunks
    chunks_set = set()
    all_passages = []
    
    # Extract passages from validation set
    for item in val_data:
        contexts = item.get("context", [])
        for title, sentences in contexts:
            passage_text = f"Title: {title}\n" + "".join(sentences)
            
            if passage_text not in chunks_set:
                chunks_set.add(passage_text)
                all_passages.append(passage_text)
                
    print(f"Total unique passages in validation set: {len(all_passages)}")
    
    # In the paper, they choose 1000 questions and their associated passages + distractors.
    # To get 10k documents, we just take 10,000 passages from the all_passages list.
    # To ensure our 1000 questions have their answers, we must make sure all passages 
    # associated with our 1000 questions are included first!
    
    required_passages = set()
    for item in val_data:
        if item['_id'] in q_ids:
            for title, sentences in item.get('context', []):
                passage_text = f"Title: {title}\n" + "".join(sentences)
                required_passages.add(passage_text)
                
    print(f"Passages required for the 1000 questions: {len(required_passages)}")
    
    final_chunks = list(required_passages)
    
    # Fill the rest up to 10,000
    remaining_passages = [p for p in all_passages if p not in required_passages]
    random.seed(42)
    random.shuffle(remaining_passages)
    
    needed_distractors = 10000 - len(final_chunks)
    if needed_distractors > 0:
        final_chunks.extend(remaining_passages[:needed_distractors])
        
    print(f"Final chunks count: {len(final_chunks)}")
    
    # Save to chunks_10k.json
    out_path = '/var/www/grasp-next/services/linear-rag/dataset/2wikimultihop/chunks_10k.json'
    with open(out_path, 'w') as f:
        json.dump(final_chunks, f, indent=2)
        
    print(f"Saved {len(final_chunks)} chunks to {out_path}")

if __name__ == "__main__":
    main()
