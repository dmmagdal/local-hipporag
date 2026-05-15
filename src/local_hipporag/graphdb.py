# graphdb.py


from typing import List, Tuple

import ladybug
from ladybug import QueryResult


class LadybugGraphDB:
	def __init__(self, db_path: str, hipporagv2: bool = False):
		# Initialize database and connection.
		self.db = ladybug.Database(db_path)
		self.conn = ladybug.Connection(self.db)
		self.hipporag2 = hipporagv2

		# Initialize tables.
		if self.hipporag2:
			self.conn.execute(
				"CREATE NODE TABLE Entity(name STRING, PRIMARY KEY (name))"
			)
			self.conn.execute(
				"CREATE NODE TABLE Passage(id STRING, text STRING, PRIMARY KEY (id))"
			)
			self.conn.execute(
				"CREATE REL TABLE CO_OCCURS(FROM Entity TO Entity)"
			)
			self.conn.execute(
				"CREATE REL TABLE CONTAINS(FROM Passage TO Entity)"
			)
		else:
			try:
				self.conn.execute(
					"CREATE NODE TABLE Entity(name STRING, PRIMARY KEY (name))"
				)
				self.conn.execute(
					"CREATE REL TABLE CO_OCCURS(FROM Entity TO Entity)"
				)
			except Exception as e: 
				print(f"Failed to create tables. Exception raised: {e}")


	def co_occurences(self, entity_1: str, entity_2: str) -> None:
		query = f"""
		MATCH (a:Entity {{name: $name1}}), (b:Entity {{name: $name2}})
		MERGE (a)-[:CO_OCCURS]->(b)
		"""
		self.conn.execute(query, {"name1": entity_1, "name2": entity_2})


	def add_text(self, text: str, doc_id: str) -> None:
			self.conn.execute(
				f"CREATE (p:Passage {{id: $id, text: $text}})",
				{"id": doc_id, "text": text}
			)


	def link_docs_to_entities(self, doc_id: str, entity: str) -> None:
		command = f"""
			MATCH (p:Passage {{id: $doc_id}}), (e:Entity {{name: $entity}})
			MERGE (p)-[:CONTAINS]->(e)
		"""
		self.conn.execute(command, {"doc_id": doc_id, "entity": entity})


	def add_entity(self, entity: str) -> None:
		self.conn.execute(f"CREATE (e:Entity {{name: '{entity}'}})")

	
	def get_topology(self, hipporagv2: bool = False) -> Tuple[QueryResult, QueryResult] | Tuple[QueryResult, QueryResult, QueryResult, QueryResult]:
		if hipporagv2:
			# We pull both Entity and Passage edges
			entities = self.conn.execute(
				"MATCH (a:Entity) RETURN a.name AS id"
			)
			passages = self.conn.execute(
				"MATCH (p:Passage) RETURN p.id AS id"
			)
			
			co_occurs = self.conn.execute(
				"MATCH (a:Entity)-[:CO_OCCURS]->(b:Entity) RETURN a.name AS src, b.name AS dst"
			)
			contains = self.conn.execute(
				"MATCH (p:Passage)-[:CONTAINS]->(e:Entity) RETURN p.id AS src, e.name AS dst"
			)
			return entities, passages, co_occurs, contains
		else:
			nodes = self.conn.execute(
				"MATCH (a:Entity) RETURN a.name"
			)
			edges = self.conn.execute(
				"MATCH (a:Entity)-[:CO_OCCURS]->(b:Entity) RETURN a.name, b.name"
			)
			return nodes, edges
		

	def get_passage(self, doc_id: str) -> str:
		return self.conn.execute(
			f"MATCH (p:Passage {{id: '{doc_id}'}}) RETURN p.text"
		).get_as_df()['p.text'][0]
	

	def add_triplet(self, source: str, target: str, relationship: str = "UNK", summary: str = "") -> None:
		self.conn.execute(
			"MATCH (a:Entity {id: $s}), (b:Entity {id: $t}) MERGE (a)-[:RELATES {relation: $r, `desc`: $d}]->(b)",
            {
				"s": source.lower(), 
				"t": target.lower(), 
				"r": relationship, 
				"d": summary
			}
		)


	def batch_co_occurences(self, pairs: List[Tuple[str, str]]) -> None:
		query = """
			UNWIND $pairs AS pair
			MATCH (a:Entity {name: pair[0]}), (b:Entity {name: pair[1]})
			MERGE (a)-[:CO_OCCURS]->(b)
		"""
		self.conn.execute(query, {"pairs": pairs})


	def query(self, entity_id: str) -> List[str]:
		return self.conn.execute(
			"MATCH (a:Entity {id: $id})-[r:RELATES]->(b) RETURN a.id, r.relation, b.id",
			{
				"id": entity_id
			}
		)

	
	def checkpoint(self) -> None:
		self.conn.execute("CHECKPOINT;")


	def close_db(self) -> None:
		self.conn.close()