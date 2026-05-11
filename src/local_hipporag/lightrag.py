# lightrag.py


import json
from pathlib import Path
from typing import List
import uuid

from local_vectors import LocalEmbedder, LanceDBConnection, detect_device
import pyarrow as pa

from .chunker import Chunker
from .graphdb import LadybugGraphDB
from .llm import OllamaLLM, GlinerLLM


class LightRAG:
	def __init__(self, 
		embed_model_id: str,
		vector_db_path: str,
		graph_db_path: str,
		llm_model: str,
		token_overlap: int = 128,													# Chunker + Embedder kwargs
		batch_size: int = 8,
		device: str = "cpu",
		use_binary: bool = False,
		query_metric: str = "cosine",
		model_save_root: str = Path.home() / ".cache" / "local-graphrag" / "models",
		host: str = "http://localhost:11434",   									# LLM kwargs
		gliner_model: str = None,
		spacy_model: str = None,
		entity_items: List[str] = None,
		summary_model: str = None,
	):
		# Initialize the text chunker.
		self.chunker = Chunker(
			model_id=embed_model_id,
			token_overlap=token_overlap,
			batch_size=batch_size,
			device=device,
			model_save_root=model_save_root,
		)

		# Initialize the text embedder.
		self.embedder = LocalEmbedder(
			model_id=embed_model_id,
			model_save_root=model_save_root,
			token_overlap=token_overlap,
			batch_size=batch_size,
			device=device,
		)

		# Initialize the vectordb.
		self.vectordb = LanceDBConnection(
			vector_db_path
		)

		# Flag whether we're using binary embeddings or full precision 
		# (as per supported on local-vectors).
		self.use_binary = use_binary

		# Initialize the graphdb.
		self.graphdb = LadybugGraphDB(
			db_path=graph_db_path,
		)

		# Initialize the LLM model(s) depending on whether "classical"
		# models have been specified. Fall back to ollama LLM if not.
		classical_models = [gliner_model, spacy_model, summary_model]
		if all(model is not None for model in classical_models):
			self.llm = GlinerLLM(
				llm_model,
				gliner_model, 
				spacy_model, 
				summary_model, 
				entity_items,
				device=device,
				model_save_root=model_save_root,
				host=host,
			)
		else:
			self.llm = OllamaLLM(
				llm_model=llm_model,
				host=host,
			)

		# Set the query metric.
		self.metric = query_metric


	def get_dims(self) -> int:
		model_metadata = self.embedder.model_metadata
		return model_metadata["binary_dims"] if self.use_binary else model_metadata["dims"]


	def build_vector_table(self, table_name: str, schema: pa.Schema) -> None:
		self.vectordb.create_table(table_name, schema)


	def set_query_metric(self, query_metric: str) -> None:
		self.metric = query_metric


	def ingest(self, text: str, doc_id: str, table_name: str) -> None:
		# Step 1: Chunk & pass the documents to the embedding.
		chunked_text = self.chunker.chunk_text(text)
		embeddings = self.embedder.embed_text(
			text, 
			truncate=False, 
			to_binary=self.use_binary
		)

		# Check that both the number of text chunks matches the number 
		# of embeddings (they should, considering they use the same
		# vector_preprocessing function from local-vectors).
		assert len(chunked_text) == len(embeddings), \
			"Expected number of embeddings generated to match the number of chunks generated for the text."

		# Error checking in case the user hasn't initialized the 
		# desired table yet.
		if table_name not in self.vectordb.table_names():
			raise ValueError(f"Table {table_name} has not yet been initialize for the vectordb. Current tables include {', '.join(self.vectordb.table_names())}")
		
		# Iterate through the data.
		vector_entries = []
		for idx, (chunk, emb) in enumerate(zip(chunked_text, embeddings)):
			chunk_id = f"{doc_id}_chunk_{idx}"
			subtext = text[chunk["text_idx"]: chunk["text_idx"] + chunk["text_len"]]

			# 1. Index the chunk.
			vector_entries.append({
				"id": chunk_id,
				"vector": emb["vector_binary"] if self.use_binary else emb["vector_full"],
				"text": subtext,
				"type": "chunk",
			})

			# 2. Extract and index graph elements.'
			entities, relationships = self.llm.extract_knowledge_graph(subtext)
			for entity in entities:
				ent_name = entity["text"].lower()
				ent_emb = self.embedder.embed_text(
					ent_name,
					truncate=True,
					to_binary=self.use_binary,
					vectors_only=True,
				)[0]

				# Embed entity for vector search.
				vector_entries.append({
					"id": f"entity_{ent_name}",
					"vector": ent_emb["vector_binary"] if self.use_binary else ent_emb["vector_full"],
					"text": f"Entity: {ent_name}", # Include entity label?
					"type": "entity",
				})

				# Store in graph.
				self.graphdb.add_entity(
					ent_name, entity["text"], entity["label"]
				)

			for relation in relationships:
				# Summarize the relationship for high level indexing.
				summary = self.llm.generate_summary(relation)
				summary_emb = self.embedder.embed_text(
					summary, 
					truncate=True,
					to_binary=self.use_binary,
					vectors_only=True,
				)[0]

				# Embed summary for vector search.
				vector_entries.append({
					"id": f"relationship_{uuid.uuid4().hex[:8]}",
					"vector": summary_emb["vector_binary"] if self.use_binary else summary_emb["vector_full"],
					"text": summary,
					"type": "relationship"
				})

				# Store in graph.
				self.graphdb.add_triplet(
					source=relation["src"], 
					target=relation["tgt"],
					relationship=relation["rel"],
					summary=relation["desc"],
				)

		# Write the vector data to the table.
		self.vectordb.update_table(
			table_name=table_name,
			data=vector_entries,
		)


	def query(self, query: str, table_name: str, top_k: int = 5) -> str:
		# Error checking in case the user hasn't initialized the 
		# desired table yet.
		if table_name not in self.vectordb.table_names():
			raise ValueError(f"Table {table_name} has not yet been initialize for the vectordb. Current tables include {', '.join(self.vectordb.table_names())}")
		
		# 1. Hybrid retrieval from vectordb (chunks, entities, & 
		# relationships).
		emb_query = self.embedder.embed_text(
			query,
			truncate=True,
			to_binary=self.use_binary,
			vectors_only=True,
		)
		results = self.vectordb.search_table(
			table_name=table_name,
			query_vector=emb_query["vector_binary"] if self.use_binary else emb_query["vector_full"],
			top_k=top_k,
			metric=self.metric
		)

		chunk_context = []
		graph_context = []

		for result in results:
			if result["type"] == "chunk":
				chunk_context.append(result["text"])
			else:
				graph_context.append(result["text"])

		# 2. Graph expansion (fetch neighboring relations for retrieved
		# entities).
		expanded_context = []
		for result in results:
			if result["type"] == "entity":
				entity_id = result['id'].replace("entity_", "")
				graph_results = self.graphdb.query(entity_id)
				while graph_results.has_next():
					neighbor = graph_results.get_next()
					expanded_context.append(
						f"{neighbor[0]} --{neighbor[1]}--> {neighbor[2]}"
					)

		# 3. Response synthesis.
		final_context = "\n".join([
			"### RELEVANT TEXT ###", *chunk_context[:3],
			"### KEY RELATIONSHIPS ###", *graph_context[:5],
			"### GRAPH STRUCTURE ###", *expanded_context[:5]
		])
		prompt = f"""Given the following multi-level context, answer the question.
		QUERY: {query}
		CONTEXT:
		{final_context}
		ANSWER:
		"""
		return self.llm.generate_response(prompt)