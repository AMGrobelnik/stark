import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from stark_qa import load_qa, load_skb
from llama_index.core import (
    StorageContext,
    Document,
    KnowledgeGraphIndex,
    Settings,
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Neo4j database connection settings
neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
neo4j_user = os.getenv("NEO4J_USER", "neo4j")
neo4j_password = os.getenv("NEO4J_PASSWORD", "test12345")

# Connect to Neo4j with the new password
driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

# Configure OpenAI LLM and embedding model
llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
Settings.llm = llm
Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
Settings.chunk_size = 512

# Load dataset
dataset_name = "amazon"
qa_dataset = load_qa(dataset_name)
skb = load_skb(dataset_name, download_processed=True)


# Function to create nodes in Neo4j
def create_nodes(tx, node_id, node_type, node_info):
    tx.run(
        "CREATE (n:Node {id: $id, type: $type, info: $info})",
        id=node_id,
        type=node_type,
        info=node_info,
    )


# Function to create relationships in Neo4j
def create_relationships(tx, src, rel, dst):
    tx.run(
        "MATCH (a:Node {id: $src}), (b:Node {id: $dst}) "
        "CREATE (a)-[:REL {type: $type}]->(b)",
        src=src,
        dst=dst,
        type=rel,
    )


# Function to clear existing data
def clear_data(tx):
    tx.run("MATCH (n) DETACH DELETE n")


# Migrate data to Neo4j
print("Starting data migration to Neo4j...")

with driver.session() as session:
    # Clear existing data
    print("Clearing existing data...")
    session.write_transaction(clear_data)

    # Create a small subset of nodes and relationships for debugging
    num_nodes = 100
    num_rels = 10

    # Create nodes
    print("Creating nodes...")
    for node_id in tqdm(range(num_nodes), desc="Nodes"):
        node_type = skb.get_node_type_by_id(node_id)
        node_info = skb.get_doc_info(node_id, add_rel=False)
        session.write_transaction(create_nodes, node_id, node_type, node_info)

    # Create relationships
    print("Creating relationships...")
    for src, rel, dst in tqdm(skb.get_tuples()[:num_rels], desc="Relationships"):
        session.write_transaction(create_relationships, src, rel, dst)

print("Data migration completed.")

# Verify data migration
with driver.session() as session:
    result = session.run("MATCH (n) RETURN count(n) as node_count")
    node_count = result.single()["node_count"]
    print(f"Number of nodes in Neo4j: {node_count}")

# Create a StorageContext that uses the Neo4j graph store
graph_store = Neo4jGraphStore(
    url=neo4j_uri, username=neo4j_user, password=neo4j_password
)
storage_context = StorageContext.from_defaults(graph_store=graph_store)

# Create documents from SKB nodes
print("Creating documents from SKB nodes...")
documents = [
    Document(text=skb.get_doc_info(node_id, add_rel=True), doc_id=str(node_id))
    for node_id in tqdm(range(num_nodes), desc="Documents")
]

# Create the knowledge graph index
print("Creating knowledge graph index...")
index = KnowledgeGraphIndex.from_documents(
    documents,
    storage_context=storage_context,
    max_triplets_per_chunk=2,
)

# Create a query engine
print("Setting up query engine...")
query_engine = index.as_query_engine(include_text=True, response_mode="tree_summarize")

# Example query using a random question from the QA dataset
print("Performing example query...")
query, q_id, answer_ids, _ = qa_dataset[0]
response = query_engine.query(query)
print("Query:", query)
print("Response:", response)
print("Actual answer IDs:", answer_ids)
print("Actual answers:", [skb[aid].title for aid in answer_ids])

# Close the Neo4j driver connection
driver.close()
print("Neo4j driver connection closed.")


# import os
# from dotenv import load_dotenv
# from neo4j import GraphDatabase
# from stark_qa import load_qa, load_skb
# from llama_index.core import (
#     StorageContext,
#     Document,
#     KnowledgeGraphIndex,
#     Settings,
# )
# from llama_index.llms.openai import OpenAI
# from llama_index.embeddings.openai import OpenAIEmbedding
# from llama_index.graph_stores.neo4j import Neo4jGraphStore

# # Load environment variables
# load_dotenv()

# # Neo4j database connection settings
# neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
# neo4j_user = os.getenv("NEO4J_USER", "neo4j")
# neo4j_old_password = os.getenv("NEO4J_OLD_PASSWORD", "neo4j")
# neo4j_new_password = os.getenv("NEO4J_PASSWORD", "test12345")

# # Change Neo4j password
# driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_old_password))


# def change_password(tx, old_password, new_password):
#     tx.run(
#         "ALTER CURRENT USER SET PASSWORD FROM $old_password TO $new_password",
#         old_password=old_password,
#         new_password=new_password,
#     )


# with driver.session(database="system") as session:
#     session.write_transaction(change_password, neo4j_old_password, neo4j_new_password)

# driver.close()

# # Connect to Neo4j with the new password
# driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_new_password))

# # Configure OpenAI LLM and embedding model
# llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
# Settings.llm = llm
# Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
# Settings.chunk_size = 512

# # Load dataset
# dataset_name = "amazon"
# qa_dataset = load_qa(dataset_name)
# skb = load_skb(dataset_name, download_processed=True)


# # Function to create nodes in Neo4j
# def create_nodes(tx, node_id, node_type, node_info):
#     tx.run(
#         "CREATE (n:Node {id: $id, type: $type, info: $info})",
#         id=node_id,
#         type=node_type,
#         info=node_info,
#     )


# # Function to create relationships in Neo4j
# def create_relationships(tx, src, rel, dst):
#     tx.run(
#         "MATCH (a:Node {id: $src}), (b:Node {id: $dst}) "
#         "CREATE (a)-[:REL {type: $type}]->(b)",
#         src=src,
#         dst=dst,
#         type=rel,
#     )


# # Function to clear existing data
# def clear_data(tx):
#     tx.run("MATCH (n) DETACH DELETE n")


# # Migrate data to Neo4j
# with driver.session() as session:
#     # Clear existing data
#     session.write_transaction(clear_data)

#     # Create nodes
#     for node_id in range(skb.num_nodes()):
#         node_type = skb.get_node_type_by_id(node_id)
#         node_info = skb.get_doc_info(node_id, add_rel=False)
#         session.write_transaction(create_nodes, node_id, node_type, node_info)

#     # Create relationships
#     for src, rel, dst in skb.get_tuples():
#         session.write_transaction(create_relationships, src, rel, dst)

# # Verify data migration
# with driver.session() as session:
#     result = session.run("MATCH (n) RETURN count(n) as node_count")
#     print("Number of nodes in Neo4j:", result.single()["node_count"])

# # Create a StorageContext that uses the Neo4j graph store
# graph_store = Neo4jGraphStore(
#     url=neo4j_uri, username=neo4j_user, password=neo4j_new_password
# )
# storage_context = StorageContext.from_defaults(graph_store=graph_store)

# # Create documents from SKB nodes
# documents = [
#     Document(text=skb.get_doc_info(node_id, add_rel=True), doc_id=str(node_id))
#     for node_id in range(skb.num_nodes())
# ]

# # Create the knowledge graph index
# index = KnowledgeGraphIndex.from_documents(
#     documents,
#     storage_context=storage_context,
#     max_triplets_per_chunk=2,
# )

# # Create a query engine
# query_engine = index.as_query_engine(include_text=True, response_mode="tree_summarize")

# # Example query using a random question from the QA dataset
# query, q_id, answer_ids, _ = qa_dataset[0]
# response = query_engine.query(query)
# print("Query:", query)
# print("Response:", response)
# print("Actual answer IDs:", answer_ids)
# print("Actual answers:", [skb[aid].title for aid in answer_ids])

# # Close the Neo4j driver connection
# driver.close()


# from llama_index.core.graph_stores import SimpleGraphStore

# import json
# import os
# from dotenv import load_dotenv
# from llama_index.core import (
#     StorageContext,
#     Document,
#     KnowledgeGraphIndex,
#     Settings,
# )
# from llama_index.llms.openai import OpenAI
# from llama_index.embeddings.openai import OpenAIEmbedding
# from stark_qa import load_qa, load_skb
# import networkx as nx

# # from llama_index.graph_stores import SimpleGraphStore
# from llama_index.graph_stores.neo4j import Neo4jGraphStore


# # Load environment variables
# load_dotenv()

# # Ensure the CUDA_VISIBLE_DEVICES variable is set correctly
# print(f"CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES')}")

# # Configure OpenAI LLM and embedding model
# Settings.llm = OpenAI(model="gpt-3.5-turbo")
# Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
# Settings.chunk_size = 512

# # Load STARK Knowledge Base and QA dataset
# dataset_name = "amazon"
# qa_dataset = load_qa(dataset_name)
# skb = load_skb(dataset_name, download_processed=True)

# # Path to persist the graph store
# persist_dir = "persist_dir"
# persist_path = os.path.join(persist_dir, "graph_store.json")

# # Check if the graph store already exists
# if os.path.exists(persist_path):
#     graph_store = SimpleGraphStore.from_persist_path(persist_path)
# else:
#     # Create a custom graph store using STARK SKB
#     graph = nx.DiGraph()
#     for node_id in range(skb.num_nodes()):
#         node_type = skb.get_node_type_by_id(node_id)
#         node_info = skb.get_doc_info(node_id, add_rel=False)
#         graph.add_node(node_id, type=node_type, info=node_info)

#     for edge in skb.get_tuples():
#         src, rel, dst = edge
#         graph.add_edge(src, dst, relation=rel)

#     # Initialize SimpleGraphStore with the graph data
#     graph_store_data = {
#         "graph_dict": {
#             node_id: [
#                 [rel, dst]
#                 for src, rel, dst in graph.edges(data="relation")
#                 if src == node_id
#             ]
#             for node_id in graph.nodes
#         }
#     }
#     graph_store = SimpleGraphStore.from_dict(graph_store_data)
#     graph_store.persist(persist_path)

# storage_context = StorageContext.from_defaults(graph_store=graph_store)

# # Create documents from SKB nodes
# documents = [
#     Document(text=skb.get_doc_info(node_id, add_rel=True), doc_id=str(node_id))
#     for node_id in range(skb.num_nodes())
# ]

# # Create the knowledge graph index
# index = KnowledgeGraphIndex.from_documents(
#     documents,
#     storage_context=storage_context,
#     max_triplets_per_chunk=2,
# )

# # Create a query engine
# query_engine = index.as_query_engine(include_text=True, response_mode="tree_summarize")

# # Example query using a random question from the QA dataset
# query, q_id, answer_ids, _ = qa_dataset[0]
# response = query_engine.query(query)
# print("Query:", query)
# print("Response:", response)
# print("Actual answer IDs:", answer_ids)
# print("Actual answers:", [skb[aid].title for aid in answer_ids])
