# storage.py

import sqlite3
import json
from typing import Dict, List, Tuple


class SQLitePassageStore: 
	def __init__(self, db_path: str):
		self.db_path = db_path
		self._init_db()
		
	
	def _init_db(self) -> None:
		with sqlite3.connect(self.db_path) as conn:
			conn.execute("""
				CREATE TABLE IF NOT EXISTS passages (
					doc_id TEXT PRIMARY KEY,
					text TEXT,
					entities TEXT
				)
			""")
			conn.commit()


	def add_passage(self, doc_id: str, text: str, entities: List[str]) -> None:
		with sqlite3.connect(self.db_path) as conn:
			conn.execute("""
				INSERT OR REPLACE INTO passages (doc_id, text, entities)
				VALUES (?, ?, ?)
			""", (doc_id, text, json.dumps(entities)))
			conn.commit()


	def add_passages_batch(self, entries: List[Tuple[str, str, List[str]]]) -> None:
		with sqlite3.connect(self.db_path) as conn:
			conn.executemany(
			"""
				INSERT OR REPLACE INTO passages (doc_id, text, entities)
				VALUES (?, ?, ?)
			""", 
			[
				(doc_id, text, json.dumps(entities)) 
				for doc_id, text, entities in entries
			]
			)
			conn.commit()


	def get_all_passages_to_entities(self) -> Dict[str, List[str]]:
		with sqlite3.connect(self.db_path) as conn:
			cursor = conn.execute(
				"SELECT doc_id, entities FROM passages"
			)
			return {
				row[0]: json.loads(row[1]) for row in cursor.fetchall()
			}


	def get_passage_text(self, doc_id: str) -> str:
		with sqlite3.connect(self.db_path) as conn:
			cursor = conn.execute(
				"SELECT text FROM passages WHERE doc_id = ?", (doc_id,)
			)
			row = cursor.fetchone()
			return row[0] if row else ""
