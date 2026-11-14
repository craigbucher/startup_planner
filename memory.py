"""Shared vector memory. Agents write findings; later agents recall them by meaning."""
# a lightweight, non-persistent vector database implementation that stores 
# documents and their embeddings directly in application memory (using a dictionary) 
# and calculates similarity scores using NumPy:
from langchain_core.vectorstores import InMemoryVectorStore

# TeamMemory is a shared memory system for agents, allowing them to write and recall information:
class TeamMemory:
    # Initialize the team memory with optional embeddings and a tracer for logging:
    def __init__(self, embeddings, tracer):
        self.tracer = tracer
        self.store = InMemoryVectorStore(embeddings) if embeddings else None
        self.fallback: list[str] = []  # used when embeddings are disabled

    # Write information to the team memory, tagging it with the agent and topic:
    def write(self, agent: str, topic: str, texts: list[str]) -> None:
        # Iterate over the texts and keep only non-empty ones:
        texts = [t for t in texts if t and t.strip()]
        if not texts:
            return
        # create tagged versions of the texts with agent and topic information:
        tagged = [f"[{agent} / {topic}] {t}" for t in texts]
        # add the tagged texts to the vector store or fallback list depending on availability:
        if self.store:
            self.store.add_texts(
                tagged, metadatas=[{"agent": agent, "topic": topic}] * len(tagged)
            )
        else:
            self.fallback.extend(tagged)
        # log the write operation with the tracer:
        self.tracer.log(
            "memory_write", agent, f"Stored {len(tagged)} {topic} item(s) in team memory"
        )
    # Recall information from the team memory based on a query, returning the most relevant items:
    def recall(self, agent: str, query: str, k: int = 4) -> list[str]:
        # Perform a similarity search in the vector store or fallback to the most recent items:
        if self.store:
            # search for the most similar items to the query in the vector store:
            hits = [d.page_content for d in self.store.similarity_search(query, k=k)]
        else:
            # fallback to the most recent items in the absence of a vector store:
            hits = self.fallback[-k:]
        # log the recall operation with the tracer:
        self.tracer.log(
            "memory_read", agent, f"Recalled {len(hits)} item(s) for: {query}", items=hits
        )
        # return the retrieved hits to the caller:
        return hits
