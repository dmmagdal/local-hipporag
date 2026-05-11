# test_data.py


import os
import random
import shutil

import datasets
from datasets import concatenate_datasets
from tqdm import tqdm


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

	# Identify the unique chunk_ids for each.
	unique_doc_ids = set()
	unique_query_doc_ids = set()

	for split, data in documents.items():
		for doc in tqdm(data, desc=f"Retrieving document IDs from {split} split in documents"):
			unique_doc_ids.add(doc["chunk_id"])

	for split, data in queries.items():
		for doc in tqdm(data, desc=f"Retrieving document IDs from {split} split in queries"):
			unique_query_doc_ids.add(doc["chunk_id"])

	if len(unique_doc_ids) >= len(unique_query_doc_ids):
		print("Number of document ids >= number of query document ids")
	else:
		print("Number of document ids < number of query document ids")

	# Compute the overlapping chunk_ids.
	overlapping_ids = unique_doc_ids.intersection(unique_query_doc_ids)
	print(f"Number of intersecting document ids: {len(overlapping_ids)}")

	# Filter out all non-overlapping chunk_ids.
	def filter_by_id(example):
		return example["chunk_id"] in overlapping_ids
	
	filtered_docs = {}
	for split, data in documents.items():
		filtered_docs[split] = data.filter(filter_by_id)
	filtered_docs_super = concatenate_datasets(filtered_docs.values())
	filtered_queries = {}
	for split, data in queries.items():
		filtered_queries[split] = data.filter(filter_by_id)
	filtered_queries_super = concatenate_datasets(filtered_queries.values())

	# Print some of the documents and queries associated to the 
	# overlapping chunk_ids.
	for id in list(overlapping_ids)[:5]:
		doc_match = filtered_docs_super.filter(lambda x: x["chunk_id"] == id)
		query_match = filtered_queries_super.filter(lambda x: x["chunk_id"] == id)

		print(f"ID: {id}")
		print(doc_match[0])
		print(query_match[0])


	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()