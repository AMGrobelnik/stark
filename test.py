# print("Hello world")
# from stark_qa import load_qa, load_skb

# dataset_name = "amazon"

# # Load the retrieval dataset
# qa_dataset = load_qa(dataset_name)
# idx_split = qa_dataset.get_idx_split()

# # Load the semi-structured knowledge base
# skb = load_skb(dataset_name, download_processed=True, root=None)
from langchain.chains import RetrievalQA

print(RetrievalQA)
