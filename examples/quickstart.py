# quickstart.py


import json
import os
import random
import shutil

import datasets
from tqdm import tqdm
import pyarrow as pa

from local_vectors import detect_device
from local_hipporag import HippoRAG, HippoRAG2


SEED = 1234
random.seed(SEED)

def main():
	# Load the dataset.
	target_dataset = "illuin-conteb/narrative-qa"
	cache_dir = f"./{target_dataset.replace('/', '_')}_cache"
	save_dir = f"./{target_dataset.replace('/', '_')}"
	subsets = ["documents", "queries"]

	# Download the dataset if it's not already available.
	if not os.path.exists(save_dir) or len(os.listdir(save_dir)) == 0:
		for subset in subsets:
			data = datasets.load_dataset(
				target_dataset,
				subset,
				cache_dir=cache_dir,
			)
			data.save_to_disk(os.path.join(save_dir, subset))

		# Clear the cache directory.
		shutil.rmtree(cache_dir)

	# Load the documents and queries.
	documents = datasets.load_from_disk(os.path.join(save_dir, "documents"))
	queries = datasets.load_from_disk(os.path.join(save_dir, "queries"))

	# Load the configuration information.
	with open("config.json", "r") as f:
		config = json.load(f)['hipporag']

	# Unpack and organized theh configuration data for each component 
	# of the hipporag.
	vector_config = config["vector"]
	graph_config = config["graph"]
	llm_config = config["llm"]

	# Clear any existing tables or databases.
	storage_artifacts = [
		vector_config["vector_db"],
		graph_config["graph_db"],
		"./graph",
		"graph.wal",
	]
	for artifact in storage_artifacts:
		if os.path.exists(artifact):
			if os.path.isdir(artifact):
				shutil.rmtree(artifact)
			else:
				os.remove(artifact)

	# Detect GPU accelerators.
	device = detect_device(force_cpu=True)

	# Initialize hipporag with the configuration.
	hipporag = HippoRAG(
		embed_model_id=vector_config["model_id"],
		vector_db_path=vector_config["vector_db"],
		graph_db_path=graph_config["graph_db"],
		llm_model=llm_config["model_id"],
		token_overlap=vector_config["token_overlap"],
		batch_size=vector_config["batch_size"],
		device=device,
		use_binary=vector_config["use_binary"],
		query_metric=vector_config["metric"],
		model_save_root=vector_config["model_save_root"],
		host=llm_config["host"],  
		gliner_model=llm_config["gliner_model"],
		spacy_model=llm_config["spacy_model"],
		summary_model=llm_config["summary_model"]
	)

	# Define schema (this is heavily dependent upon the datasets) and
	# pass that to the hipporag so that the vectordb can build the 
	# table.
	schema = pa.schema([
		pa.field("entity_name", pa.string()),
		pa.field("vector", pa.list_(
			pa.uint8() if vector_config["use_binary"] else pa.float32(), 
			hipporag.get_dims()
		)),
		pa.field("type", pa.string())
	])
	schema2 = pa.schema([
		pa.field("text", pa.string()),
		pa.field("node_id", pa.string()),
		pa.field("vector", pa.list_(
			pa.uint8() if vector_config["use_binary"] else pa.float32(), 
			hipporag.get_dims()
		)),
		pa.field("type", pa.string())
	])
	hipporag.build_vector_table(
		table_name=vector_config["table_name"],
		schema=schema
	)

	# Ingest and index the documents to the hipporag.
	for split_name, data in documents.items():
		for doc in tqdm(data, desc=f"Ingesting {split_name} split into Graph RAG"):
			hipporag.ingest(
				text=doc["chunk"], 
				doc_id=doc["chunk_id"],
				table_name=vector_config["table_name"]
			)

	# Perform a query on the hipporag.
	sampled_queries = queries.shuffle(seed=SEED).select(range(5))
	for query in sampled_queries:
		question, chunk_id, answer = query["og_query"], query["chunk_id"], query["answer"]
		hipporag_answer = hipporag.query(question)

		# Output results.
		print(f"Question: {question}")
		print(f"Expected answer: {answer} (chunk id {chunk_id})")
		print(f"Generated answer: {hipporag_answer}")
		print("-" * 72)

	# Clear all tables or databases since we're done.
	for artifact in storage_artifacts:
		if os.path.exists(artifact):
			if os.path.isdir(artifact):
				shutil.rmtree(artifact)
			else:
				os.remove(artifact)

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()