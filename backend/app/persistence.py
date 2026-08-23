"""Optional Neo4j Community projection; the domain remains in-memory by default."""
from __future__ import annotations

import json
import os
from .core import AHMemory


class Neo4jAdapter:
    def __init__(self, uri: str | None = None, user: str | None = None, password: str | None = None):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        self.driver = None

    def connect(self):
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise RuntimeError("install optional dependency with: uv sync --extra neo4j") from exc
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        return self

    def close(self):
        if self.driver: self.driver.close()

    def persist(self, memory: AHMemory) -> int:
        if self.driver is None: self.connect()
        dump = memory.export()["dump"]
        with self.driver.session() as session:
            def tx_run(tx):
                tx.run("MERGE (r:AHRevision {revision:$revision}) SET r.schema=$schema", revision=dump["revision"], schema=dump["schema"])
                for section in ("S", "C", "P", "H"):
                    for item in dump[section]:
                        label = "FirstOrderSymbol" if section == "S" else "MemoryElement"
                        tx.run(f"MERGE (n:AHObject:{label} {{uid:$uid}}) SET n.section=$section, n.payload=$payload, n.revision=$revision", uid=item["uid"], section=section, payload=json.dumps(item, ensure_ascii=False), revision=dump["revision"])
                for item in dump["L"]:
                    tx.run("MERGE (n:AHObject:AssociativeLink {uid:$uid}) SET n.section='L', n.payload=$payload, n.revision=$revision", uid=item["uid"], section="L", payload=json.dumps(item, ensure_ascii=False), revision=dump["revision"])
            session.execute_write(tx_run)
        return dump["revision"]

    def latest_revision(self) -> int:
        if self.driver is None: self.connect()
        with self.driver.session() as session:
            record = session.run("MATCH (r:AHRevision) RETURN max(r.revision) AS revision").single()
        return int(record["revision"] or 0) if record else 0

    def load(self) -> AHMemory:
        """Reconstruct the complete typed aggregate from the latest revision."""
        if self.driver is None: self.connect()
        revision = self.latest_revision()
        with self.driver.session() as session:
            rows = session.run("MATCH (n:AHObject) WHERE n.revision=$revision RETURN n.section AS section, n.payload AS payload", revision=revision)
            dump = {"schema": "AH-2026-executable-v1", "revision": revision, "S": [], "C": [], "P": [], "H": [], "L": []}
            for row in rows:
                section = row["section"]; item = json.loads(row["payload"])
                dump[section].append(item)
        return AHMemory.from_export(dump)
