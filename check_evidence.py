import json

with open('/tmp/linear-rag-dataset/2wikimultihop/chunks.json', 'r') as f:
    chunks = json.load(f)

with open('/tmp/linear-rag-dataset/2wikimultihop/questions.json', 'r') as f:
    questions = json.load(f)

# Combine all chunks into one massive string to check if answers even exist
all_text = " ".join(chunks).lower()

possible_to_answer = 0
for q in questions:
    ans = str(q['answer']).lower()
    if ans in all_text:
        possible_to_answer += 1

print(f"Total Questions: {len(questions)}")
print(f"Questions whose answer exists in chunks: {possible_to_answer} ({possible_to_answer/len(questions):.2%})")
