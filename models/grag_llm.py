from llama_index.llms.openai import OpenAI

resp = OpenAI().complete("Paul Graham is ")
print(resp)

import os
import logging
from dotenv import load_dotenv
from neo4j import GraphDatabase
from llama_index.core import (
    StorageContext,
    Settings,
    SimpleKeywordTableIndex,
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import KnowledgeGraphRAGRetriever

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("llama_index")

# Load environment variables
load_dotenv()

# Neo4j database connection settings
neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
neo4j_user = os.getenv("NEO4J_USER", "neo4j")
neo4j_password = os.getenv("NEO4J_PASSWORD", "test12345")

# OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Connect to Neo4j
driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

# Configure OpenAI LLM and embedding model
llm = OpenAI(temperature=0, model="gpt-3.5-turbo", api_key=openai_api_key)
Settings.llm = llm
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-ada-002", api_key=openai_api_key
)
Settings.chunk_size = 512
Settings.num_output = 1024  # Increase output token limit
Settings.context_window = 3900  # Increase context window


# Function to run Neo4j query and return results
def run_query(query):
    with driver.session() as session:
        result = session.run(query)
        return [record for record in result]


# Test Neo4j connection and retrieve sample data
def test_neo4j_connection():
    try:
        with driver.session() as session:
            result = session.run("MATCH (n) RETURN COUNT(n) as node_count")
            node_count = result.single()["node_count"]
            print(f"Successfully connected to Neo4j. Total nodes: {node_count}")

            # Test query to retrieve some data
            result = session.run(
                "MATCH (n) WHERE n.info IS NOT NULL RETURN n.id, n.type, n.info LIMIT 5"
            )
            records = result.data()
            if records:
                print("Sample data:")
                for record in records:
                    print(f"ID: {record['n.id']}")
                    print(f"Type: {record['n.type']}")
                    print(f"Info: {record['n.info'][:100]}...")  # First 100 characters
                    print()
            else:
                print("No nodes found with 'info' property")
    except Exception as e:
        print(f"Error connecting to Neo4j: {str(e)}")


# Run the connection test
test_neo4j_connection()

# Create a StorageContext that uses the Neo4j graph store
try:
    graph_store = Neo4jGraphStore(
        url=neo4j_uri, username=neo4j_user, password=neo4j_password
    )
    storage_context = StorageContext.from_defaults(graph_store=graph_store)
    print("Successfully created Neo4j graph store and storage context.")
except Exception as e:
    print(f"Error creating Neo4j graph store: {str(e)}")
    raise


# Function to extract entities from the info field
def extract_entities(info):
    lines = info.split("\n")
    entities = []
    for line in lines:
        if ":" in line:
            key, value = line.split(":", 1)
            entities.append(key.strip())
            entities.append(value.strip())
    return entities


# Set up RAG query engine
print("Setting up RAG query engine...")
try:
    graph_rag_retriever = KnowledgeGraphRAGRetriever(
        storage_context=storage_context,
        verbose=True,
        entity_extract_fn=extract_entities,
        max_intermediate_results=100,
        retriever_mode="keyword",
        logger=logger,
    )

    query_engine = RetrieverQueryEngine.from_args(
        graph_rag_retriever, verbose=True, logger=logger
    )
    print("Successfully set up RAG query engine.")
except Exception as e:
    print(f"Error setting up RAG query engine: {str(e)}")
    raise

# Perform example queries
print("\nPerforming example queries:")
example_queries = [
    "What products are available in the database?",
    "List some of the brands mentioned in the product information.",
    "What features are commonly mentioned in the product descriptions?",
    "Summarize some customer reviews for products in the database.",
    "What are the price ranges for the products?",
]

for query in example_queries:
    print(f"\nQuery: {query}")
    try:
        response = query_engine.query(query)
        print(f"Response: {response}")
        if response.source_nodes:
            print("Source nodes:")
            for node in response.source_nodes:
                print(f"  Node ID: {node.node.id_}")
                print(f"  Node Text: {node.node.text[:100]}...")  # First 100 characters
        else:
            print("No source nodes found.")
    except Exception as e:
        print(f"Error executing query: {str(e)}")

# Close the Neo4j driver connection
driver.close()
print("\nNeo4j driver connection closed.")


# import os
# from dotenv import load_dotenv
# from neo4j import GraphDatabase
# from stark_qa import load_qa, load_skb
# from llama_index.core import (
#     StorageContext,
#     Settings,
# )
# from llama_index.llms.openai import OpenAI
# from llama_index.embeddings.openai import OpenAIEmbedding
# from llama_index.graph_stores.neo4j import Neo4jGraphStore
# from llama_index.core.query_engine import RetrieverQueryEngine
# from llama_index.core.retrievers import KnowledgeGraphRAGRetriever
# from tqdm import tqdm

# # Load environment variables
# load_dotenv()

# # Neo4j database connection settings
# neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
# neo4j_user = os.getenv("NEO4J_USER", "neo4j")
# neo4j_password = os.getenv("NEO4J_PASSWORD", "test12345")

# # Connect to Neo4j
# driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

# # Configure OpenAI LLM and embedding model
# llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
# Settings.llm = llm
# Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
# Settings.chunk_size = 512

# # Load dataset
# dataset_name = "amazon"
# qa_dataset = load_qa(dataset_name)
# skb = load_skb(dataset_name, download_processed=True)

# # Set the limit for number of nodes to generate
# NODE_LIMIT = 1000

# # Function to create nodes in Neo4j
# def create_nodes(tx, node_id, node_type, node_info):
#     tx.run(
#         "MERGE (n:Node {id: $id}) "
#         "SET n.type = $type, n.info = $info",
#         id=node_id,
#         type=node_type,
#         info=node_info,
#     )

# # Function to create relationships in Neo4j
# def create_relationship(tx, src_id, dst_id, rel_type):
#     tx.run(
#         "MATCH (a:Node {id: $src_id}), (b:Node {id: $dst_id}) "
#         "MERGE (a)-[:REL {type: $rel_type}]->(b)",
#         src_id=src_id,
#         dst_id=dst_id,
#         rel_type=rel_type,
#     )

# # Function to clear existing data
# def clear_data(tx):
#     tx.run("MATCH (n) DETACH DELETE n")

# # Function to check if Neo4j graph needs to be generated
# def check_neo4j_graph(session):
#     result = session.run("MATCH (n) RETURN count(n) as node_count")
#     neo4j_node_count = result.single()["node_count"]
#     skb_node_count = min(NODE_LIMIT, skb.num_nodes()) if NODE_LIMIT != -1 else skb.num_nodes()
#     return neo4j_node_count == skb_node_count

# # Migrate data to Neo4j if necessary
# with driver.session() as session:
#     if not check_neo4j_graph(session):
#         print("Generating Neo4j graph...")

#         # Clear existing data
#         print("Clearing existing data...")
#         session.write_transaction(clear_data)

#         # Create nodes
#         print("Creating nodes...")
#         total_nodes = skb.num_nodes()
#         num_nodes = total_nodes if NODE_LIMIT == -1 else min(NODE_LIMIT, total_nodes)
#         for node_id in tqdm(range(num_nodes), desc="Nodes"):
#             node_type = skb.get_node_type_by_id(node_id)
#             node_info = skb.get_doc_info(node_id, add_rel=False)
#             session.write_transaction(create_nodes, node_id, node_type, node_info)

#         # Create relationships
#         print("Creating relationships...")
#         edge_index = skb.edge_index
#         edge_types = skb.edge_types
#         num_edges = edge_index.shape[1]
#         for i in tqdm(range(num_edges), desc="Relationships"):
#             src_id = int(edge_index[0, i])
#             dst_id = int(edge_index[1, i])
#             if src_id < num_nodes and dst_id < num_nodes:
#                 rel_type = skb.get_edge_type_by_id(int(edge_types[i]))
#                 session.write_transaction(create_relationship, src_id, dst_id, rel_type)

#         print("Neo4j graph generation completed.")
#     else:
#         print("Neo4j graph already exists and matches SKB. Skipping generation.")

# # Verify data migration
# print("\nVerifying data migration:")
# with driver.session() as session:
#     # Check node count
#     result = session.run("MATCH (n) RETURN count(n) as node_count")
#     node_count = result.single()["node_count"]
#     print(f"Number of nodes in Neo4j: {node_count}")

#     # Check relationship count
#     result = session.run("MATCH ()-[r:REL]->() RETURN count(r) as rel_count")
#     rel_count = result.single()["rel_count"]
#     print(f"Number of relationships in Neo4j: {rel_count}")

# # Create a StorageContext that uses the Neo4j graph store
# graph_store = Neo4jGraphStore(
#     url=neo4j_uri, username=neo4j_user, password=neo4j_password
# )
# storage_context = StorageContext.from_defaults(property_graph_store=graph_store)

# # Set up RAG query engine
# print("Setting up RAG query engine...")
# graph_rag_retriever = KnowledgeGraphRAGRetriever(
#     storage_context=storage_context,
#     verbose=True,
# )

# query_engine = RetrieverQueryEngine.from_args(
#     graph_rag_retriever,
# )

# # Perform example queries
# print("\nPerforming example queries:")
# example_queries = [
#     "What are the best-selling products?",
#     "Tell me about the most popular brands.",
#     "What categories of products are available?",
# ]

# for query in example_queries:
#     print(f"\nQuery: {query}")
#     response = query_engine.query(query)
#     print(f"Response: {response}")

# # Close the Neo4j driver connection
# driver.close()
# print("\nNeo4j driver connection closed.")
