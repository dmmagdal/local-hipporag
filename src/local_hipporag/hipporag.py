# hipporag.py


import json
from pathlib import Path
from typing import List

from local_vectors import LocalEmbedder, LanceDBConnection, detect_device
import networkx as nx
import pyarrow as pa

from .chunker import Chunker
from .graphdb import LadybugGraphDB
from .llm import OllamaLLM, GlinerLLM


class HippoRAG:
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
		
		# Map of documents and documents to entities.
		self.passages_db = {}
		self.passages_to_entities = {}


	def get_dims(self) -> int:
		model_metadata = self.embedder.model_metadata
		return model_metadata["binary_dims"] if self.use_binary else model_metadata["dims"]


	def build_vector_table(self, table_name: str, schema: pa.Schema) -> None:
		self.vectordb.create_table(table_name, schema)


	def set_query_metric(self, query_metric: str) -> None:
		self.metric = query_metric


	def ingest(self, text: str, doc_id: str, table_name: str) -> None:
		self.passages_db[doc_id] = text
		
		# Extract entities with GLiNER.
		entities = list(set([
			entity['text'].lower() for entity in self.llm.extract_entities(text)
		]))
		self.passages_to_entities[doc_id] = entities
		
		if not entities:
			return 
		
		vector_data = []
		for entity in entities:
			# Insert node into graphdb (ignore if it already exists).
			try:
				self.graphdb.add_entity(entity)
				embedding = self.embedder.embed_text(
					entity,
					truncate=True,
					to_binary=self.use_binary,
					vectors_only=True
				)[0]
				vector_data.append({
					"vector": embedding["vector_binary"] if self.use_binary else embedding["vector_full"], 
					"entity_name": entity
				})
			except:
				pass

		# Insert embeddings into vectordb.
		if vector_data:
			self.vectordb.update_table(
				table_name=table_name, 
				data=vector_data
			)

		# Create unweighted co-occurence edges.
		for i in range(len(entities)):
			for j in range(i + 1, len(entities)):
				self.graphdb.co_occurences(entities[i], entities[j])


	def query(self, query: str, table_name: str, top_k: int = 5) -> str:
		# Error checking in case the user hasn't initialized the 
		# desired table yet.
		if table_name not in self.vectordb.table_names():
			raise ValueError(f"Table {table_name} has not yet been initialize for the vectordb. Current tables include {', '.join(self.vectordb.table_names())}")
		
		# Extract the query entities.
		query_entities = list(set([
			entity["text"].lower() 
			for entity in self.llm.extract_entities(query)
		]))

		# Semantic linking (bridge vocabulary gaps using lancedb).
		seed_nodes = []
		for query_entity in query_entities:
			query_embedding = self.embedder.embed_text(
				query_entity,
				truncate=True,
				to_binary=self.use_binary,
				vectors_only=True,
			)
			result = self.vectordb.search_table(
				table_name=table_name,
				query_vector=query_embedding["vector_binary"] if self.use_binary else query_embedding["vector_full"],
				top_k=1,
				metric=self.metric
			)

			if result:
				seed_nodes.append(result[0]["entity_name"])

		if not seed_nodes:
			retrieved_context = []
		
		# Spreading activation (hippocampal retrieval via ladybugdb &
		# networkx). Pull the graph topology.
		nodes, edges = self.graphdb.get_topology()
		nodes_df, edges_df = nodes.get_as_df(), edges.get_as_df()

		nx_graph = nx.Graph()
		nx_graph.add_nodes_from(nodes_df["a.name"])
		for _, row in edges_df.iterrows():
			nx_graph.add_edge(row["a.name"], row["b.name"])

		# Define restart probabilities focused entirely on the seed 
		# nodes.
		personalization = {node: 0.0 for node in nx_graph.nodes()}
		for seed in seed_nodes:
			if seed in personalization:
				personalization[seed] = 1.0

		# Run personalized page rank.
		ppr_scores = nx.pagerank(
			nx_graph, 
			alpha=0.85, 
			personalization=personalization
		)

		# Rank passages based on total activation of their contained 
		# entities.
		passage_scores = {}
		for passage_id, passage_entities in passage_entities.items():
			passage_scores[passage_id] = sum(
				ppr_scores.get(entity, 0) for entity in passage_entities
			)

		ranked_passage_ids = sorted(
			passage_scores, key=passage_scores.get, reverse=True
		)[:top_k]
		retrieved_context = [
			self.passages_db[passage_id] 
			for passage_id in ranked_passage_ids
		]

		# Response synthesis.
		final_context = "\n".join(retrieved_context)
		prompt = f"""Given the following multi-level context, answer the question.
		QUERY: {query}
		CONTEXT:
		{final_context}
		ANSWER:
		"""
		return self.llm.generate_response(prompt)


class HippoRAG2:
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
			hipporagv2=True
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
		
		# Map of documents and documents to entities.
		self.passages_db = {}
		self.passages_to_entities = {}


	def get_dims(self) -> int:
		model_metadata = self.embedder.model_metadata
		return model_metadata["binary_dims"] if self.use_binary else model_metadata["dims"]


	def build_vector_table(self, table_name: str, schema: pa.Schema) -> None:
		self.vectordb.create_table(table_name, schema)


	def set_query_metric(self, query_metric: str) -> None:
		self.metric = query_metric


	def ingest(self, text: str, doc_id: str, table_name: str) -> None:
		# Extract entities with GLiNER.
		entities = list(set([
			entity["text"].lower() for entity in self.llm.extract_entities(text)
		]))
		self.passages_to_entities[doc_id] = entities
		
		if not entities:
			return
		
		# Insert document into ladybug.
		self.graphdb.add_text(text, doc_id=doc_id)

		# Store document embeddings into lancedb.
		p_vectors = [
			{
				"vector": vector["vector_binary"] if self.use_binary else vector["vector_full"],
				"node_id": doc_id,
				"type": "passage",
				"text": text[vector["text_idx"]: vector["text_idx"] + vector["text_len"]]
			}
			for vector in self.embedder.embed_text(
				text,
				to_binary=self.use_binary,
			)
		]

		e_vectors = []
		for entity in entities:
			# Insert entity node.
			try:
				self.graphdb.add_entity(entity)
				entity_embedding = self.embedder.embed_text(
					entity, 
					truncate=True, 
					to_binary=self.use_binary,
					vectors_only=True
				)
				e_vectors.append({
					"vector": entity_embedding["vector_binary"] if self.use_binary else entity_embedding["vector_full"],
					"node_id": entity,
					"type": "entity",
					"text": ""
				})
			except:
				pass

			# Link documents to entities.
			self.graphdb.link_docs_to_entities(doc_id, entity)

		# Create unweighted co-occurence edges.
		for i in range(len(entities)):
			for j in range(i + 1, len(entities)):
				self.graphdb.co_occurences(entities[i], entities[j])

		# Insert embeddings into vectordb.
		vector_data = p_vectors + e_vectors
		self.vectordb.update_table(
			table_name=table_name, 
			data=vector_data
		)


	def query(self, query: str, table_name: str, top_k: int = 5) -> str:
		# Deep contextualization (embed query).
		query_embedding = self.embedder.embed_text(
			query,
			truncate=True,
			to_binary=self.use_binary,
			vectors_only=True
		)

		# Find seed nodes (can be entities or passages/documents).
		results = self.vectordb.search_table(
			table_name=table_name,
			query_vector=query_embedding["vector_binary"] if self.use_binary else query_embedding["vector_full"],
			top_k=top_k,
			metric=self.metric,
		)
		seed_nodes = [result["node_id"] for result in results]
		
		if not seed_nodes:
			retrieved_context = []

		# Build graph topology for page rank. Pull the entities and
		# passages/documents edges.
		entities, passages, co_occurs, contains = self.graphdb.get_topology(hipporagv2=True)
		entities_df, passages_df, co_occurs_df, contains_df = entities.get_as_df(), passages.get_as_df(), co_occurs.get_as_df(), contains.get_as_df()

		nx_graph = nx.Graph()
		nx_graph.add_edges_from(entities_df["id"])
		nx_graph.add_edges_from(passages_df["id"])
		for _, row in co_occurs_df.iterrows():
			nx_graph.add_edge(row["src"], row["dst"])
		for _, row in contains_df.iterrows():
			nx_graph.add_edge(row["src"], row["dst"])

		# Define restart probabilities focused entirely on the seed 
		# nodes.
		personalization = {node: 0.0 for node in nx_graph.nodes()}
		for seed in seed_nodes:
			if seed in personalization:
				personalization[seed] = 1.0

		# Run personalized page rank.
		ppr_scores = nx.pagerank(
			nx_graph, 
			alpha=0.85, 
			personalization=personalization
		)

		# Rank passages nodes directly from PageRank scores.
		passage_scores = {
			passage_id: ppr_scores.get(passage_id, 0)
			for passage_id in passages_df["id"]
		}

		ranked_passage_ids = sorted(
			passage_scores, key=passage_scores.get, reverse=True
		)[:top_k]
		retrieved_context = [
			(passage_id, self.graphdb.get_passage(passage_id))
			for passage_id in ranked_passage_ids
		]

		filtered_texts = self.llm.filter_texts(self.query, retrieved_context)
		retrieved_context = [text for _, text in filtered_texts]

		# Response synthesis.
		final_context = "\n".join(retrieved_context)
		prompt = f"""Given the following multi-level context, answer the question.
		QUERY: {query}
		CONTEXT:
		{final_context}
		ANSWER:
		"""
		return self.llm.generate_response(prompt)