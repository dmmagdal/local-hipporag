# llm.py
# This file contains functions for interacting with LLMs via Ollama.


import json
import os
from pathlib import Path
import shutil
from typing import List, Dict, Tuple

from gliner import GLiNER
import requests
import spacy
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
from transformers import AutoModelForCausalLM


class LLM:
	def __init__(
		self, 
		llm_model: str,
	):
		self.LLM_MODEL = llm_model


	def call_llm(self, prompt: str) -> str:
		return ""
	

	def extract_triplets(self, text: str) -> List[Dict[str, str]]:	
		return []
		

	def extract_entities(self, text: str) -> List[str]:
		return []
	

	def generate_response(self, prompt: str) -> str:
		return ""


class OllamaLLM(LLM):
	def __init__(
		self, 
		llm_model: str,
		host: str = "http://localhost:11434", 
	):
		self.LLM_MODEL = llm_model
		self.OLLAMA_URL = host.rstrip("/")


	def call_llm(self, prompt: str, format: str = "") -> str:
		payload = {
			"model": self.LLM_MODEL, 
			"prompt": prompt, 
			"stream": False, 
		}
		if format != "":
			payload["format"] = format

		res = requests.post(
			f"{self.OLLAMA_URL}/api/generate", 
			json=payload
		)
		res.raise_for_status()
		return res.json()['response']
	

	def extract_triplets(self, text: str) -> List[Dict[str, str]]:
		# prompt = f"""Extract entities and relationships from the following text as a JSON list of objects.
		# Format: [{{"subject": "name", "relation": "description", "object": "name", "type": "category"}}]
		# Text: {text}
		# JSON:"""
		prompt = f"""You are an expert data extraction algorithm. Your task is to extract an exhaustive list of entities and their relationships from the given text.
		
		RULES:
		1. You MUST extract as many meaningful subject-relation-object triplets as possible.
		2. You MUST respond ONLY with a valid JSON array of objects.
		3. Each object MUST have exactly these four keys: "subject", "relation", "object", "type". Do not add any other keys.
		4. The "type" should be a broad category for the subject (e.g., "person", "location", "organization").

		EXAMPLE INPUT:
		John Smith works at Google. He lives in New York with his dog, Max.
		
		EXAMPLE OUTPUT:
		[
		  {{"subject": "John Smith", "relation": "works at", "object": "Google", "type": "person"}},
		  {{"subject": "John Smith", "relation": "lives in", "object": "New York", "type": "person"}},
		  {{"subject": "John Smith", "relation": "owns", "object": "Max", "type": "person"}},
		  {{"subject": "Max", "relation": "is a", "object": "dog", "type": "animal"}}
		]

		Text: {text}
		JSON:"""
		raw_json = self.call_llm(prompt, format="json")
		try: 
			return json.loads(raw_json)
		except: 
			return []


	def entity_extraction(self, text: str) -> List[Dict[str, str]]:
		prompt = f"""Extract entities from the following text as a JSON list of objects.
		Format: [{{"entity": "name"}}]
		Text: {text}
		JSON:"""
		raw_json = self.call_llm(prompt, format="json")
		try: 
			return json.loads(raw_json)
		except: 
			return []
	

	def filter_texts(self, query: str, texts: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
		valid_passages = []
		for doc_id, text in texts:
			prompt = f"Query: {query}\nPassage: {text}\nIs this passage relevant to answering the query? Answer strictly 'Yes' or 'No'."
			response = self.call_llm(prompt)
			if "yes" in response:
				valid_passages.append((doc_id, text))
				
		return valid_passages if valid_passages else texts
		

	def generate_response(self, prompt: str) -> str:
		return self.call_llm(prompt)
	

class GlinerLLM(LLM):
	def __init__(self, 
		llm_model: str,
		gliner_model: str, 
		spacy_model: str, 
		summary_model: str, 
		entity_items: List[str] = None,
		device: str = "cpu",
		model_save_root: str = Path.home() / "local_lightrag" / "models",
		host: str = None,
	):
		self.LLM_MODEL = llm_model
		self.gliner_model = gliner_model
		self.spacy_model = spacy_model
		self.summary_model = summary_model
		self.model_save_root = model_save_root
		self.device = device

		# Initialize/load models.
		self.ner = self._load_gliner_model()
		self.nlp = spacy.load(spacy_model)

		default_items = [
			"Person", "Org", "Product", "Event", "Concept"
		]
		self.entity_items = entity_items if entity_items is not None else default_items

		# Host for the LLM. Could be ollama or if not specified, just 
		# load with huggingface transformers.
		self.OLLAMA_URL = host
		if host is not None:
			self.OLLAMA_URL = self.OLLAMA_URL.rstrip("/")
			self.llm_tokenizer, self.llm_model = None, None
		else:
			self.llm_tokenizer, self.llm_model = self._load_llm_model()


	def call_llm(self, prompt: str, format: str = "") -> str:
		payload = {
			"model": self.LLM_MODEL, 
			"prompt": prompt, 
			"stream": False, 
		}
		if format != "":
			payload["format"] = format

		res = requests.post(
			f"{self.OLLAMA_URL}/api/generate", 
			json=payload
		)
		res.raise_for_status()
		return res.json()['response']


	def extract_entities(self, text) -> List[str]:
		return self.ner.predict_entities(text, self.entity_items)
	

	def extract_triplets(self, text) -> List[Dict[str, str]]:
		doc = self.nlp(text)
		relations = []

		for token in doc:
			if token.pos_ == "VERB":
				subj = [w for w in token.lefts if w.dep_ in ('nsubj', 'nsubjpass')]
				obj = [w for w in token.rights if w.dep_ in ('dobj', 'pobj', 'attr')]
				if subj and obj:
					relations.append({
						"src": subj[0].text, "tgt": obj[0].text, "rel": token.lemma_,
						"desc": f"{subj[0].text} {token.text} {obj[0].text} in the context of: {text[:50]}..."
					})
	
		return relations
	

	def filter_texts(self, query: str, texts: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
		valid_passages = []
		for doc_id, text in texts:
			prompt = f"Query: {query}\nPassage: {text}\nIs this passage relevant to answering the query? Answer strictly 'Yes' or 'No'."
			response = self.call_llm(prompt)
			if "yes" in response:
				valid_passages.append((doc_id, text))
				
		return valid_passages if valid_passages else texts
	
		
	def generate_response(self, prompt: str) -> str:
		if self.OLLAMA_URL is None:
			pass
		return self.call_llm(prompt)

		
	def _load_gliner_model(self) -> GLiNER:
		model_path = str(Path(self.model_save_root) / self.gliner_model.replace("/", "_"))

		# Check for path and that path is a directory.
		if not os.path.exists(model_path) or not os.path.isdir(model_path):
			os.makedirs(model_path, exist_ok=True)

		# Check for path to be populated with files.
		if len(os.listdir(model_path)) == 0:
			print(f"Model {self.gliner_model} needs to be downloaded.")

			# Connectivity check.
			try:
				response = requests.get("https://huggingface.co/", timeout=5)
				if response.status_code != 200:
					raise ConnectionError
			except Exception:
				print(f"Unable to reach Hugging Face to download {self.gliner_model}.")
				exit(1)

			# Create temporary cache path
			cache_path = model_path + "_tmp"
			os.makedirs(cache_path, exist_ok=True)

			# 1. Download/Load model into temporary cache. GLiNER uses 
			# HF's cache_dir internally.
			model = GLiNER.from_pretrained(
				self.gliner_model,
				cache_dir=cache_path,
				load_tokenizer=True,
			)

			# 2. Save the model to the final destination. This saves 
			# the config, model weights, and tokenizer files.
			model.save_pretrained(model_path)

			# 3. Clean up the temporary cache.
			shutil.rmtree(cache_path)
		
		# 4. Load the model from the local path. Wet 
		# local_files_only=True to ensure it doesn't try to ping HF 
		# again.
		model = GLiNER.from_pretrained(
			model_path,
			local_files_only=True
		)

		# Return the model.
		return model
	

	def _load_llm_model(self) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
		# Check for the local copy of the model. If the model doesn't have
		# a local copy (the path doesn't exist), download it.
		model_path = str(Path(self.model_save_root) / self.LLM_MODEL.replace("/", "_"))
		
		# Check for path and that path is a directory. Make it if either is
		# not true.
		if not os.path.exists(model_path) or not os.path.isdir(model_path):
			os.makedirs(model_path, exist_ok=True)

		# Check for path the be populated with files (weak check). Download
		# the tokenizer and model and clean up files once done.
		if len(os.listdir(model_path)) == 0:
			print(f"Model {self.LLM_MODEL} needs to be downloaded.")

			# Check for internet connection (also checks to see that
			# huggingface is online as well). Exit if fails.
			response = requests.get("https://huggingface.co/")
			if response.status_code != 200:
				print(f"Request to huggingface.co returned unexpected status code: {response.status_code}")
				print(f"Unable to download {self.LLM_MODEL} model.")
				exit(1)

			# Create cache path folders.
			cache_path = str(Path(self.model_save_root) / self.LLM_MODEL.replace("/", "_")) + "_tmp"
			os.makedirs(cache_path, exist_ok=True)
			os.makedirs(model_path, exist_ok=True)

			# Load tokenizer and model.
			tokenizer = AutoTokenizer.from_pretrained(
				self.LLM_MODEL, 
				cache_dir=cache_path, 
				device_map=self.device
			)
			model = AutoModelForCausalLM.from_pretrained(
				self.LLM_MODEL, 
				cache_dir=cache_path, 
				device_map=self.device,
				trust_remote_code=True, 
				use_safetensors=True
			)

			# Load the model metadata and save it to the save path.
			AutoConfig.from_pretrained(
				self.summary_model, 
				cache_dir=model_path
			)

			# Save the tokenizer and model to the save path.
			tokenizer.save_pretrained(model_path)
			model.save_pretrained(model_path)

			# Delete the cache.
			shutil.rmtree(cache_path)
		
		# Load the tokenizer and model.
		tokenizer = AutoTokenizer.from_pretrained(
			model_path, 
			device_map=self.device
		)
		model = AutoModelForCausalLM.from_pretrained(
			model_path, 
			device_map=self.device,
			trust_remote_code=True, 
			use_safetensors=True
		)

		# Return the tokenizer and model.
		return tokenizer, model