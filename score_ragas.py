import json
import os
import sys
import warnings

warnings.filterwarnings('ignore')

from ragas import evaluate
from ragas.metrics import faithfulness, context_precision, context_recall
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from datasets import Dataset

def run():
    with open("evals/multihop_hotpotqa_results.json") as f:
        data = json.load(f)
    
    details = data["modes"]["hybrid_ppr"]["details"]
    
    # We need to construct the dataset for ragas!
    # Wait, the details don't contain contexts texts! They only contain `contexts` as an integer (length).
    print("Cannot run ragas on existing json because contexts texts are missing.")
    
run()
