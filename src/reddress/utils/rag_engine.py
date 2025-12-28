import os
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime
from rich.console import Console
import numpy as np

from .document_loader import DocumentLoader
from .chunking import SmartChunker
from .embeddings import EmbeddingGenerator
from .rag_database import RAGDatabase

console = Console()


class RAGEngine:
    """Main RAG engine for indexing and querying documents"""
    
    def __init__(self, vector_store: str = "chromadb", embedding_model: str = "text-embedding-3-small"):
        self.vector_store_type = vector_store
        self.embedding_model = embedding_model
        self.vector_store = None
        self.collection = None
        self.db = RAGDatabase()
        self.use_memory = False
        self.memory_store = {}
        
        # Initialize vector store client
        if vector_store == "memory":
            self.use_memory = True
            console.print("[green]✅ Using in-memory storage (no Docker needed)[/green]")
        else:
            try:
                self._init_vector_store()
            except Exception as e:
                console.print(f"[yellow]⚠️  Vector store not available: {str(e)}[/yellow]")
                console.print("[cyan]💡 Falling back to in-memory storage[/cyan]")
                self.use_memory = True
    
    def _init_vector_store(self):
        """Initialize connection to vector store"""
        if self.vector_store_type == "chromadb":
            try:
                import chromadb
                self.vector_store = chromadb.HttpClient(host="localhost", port=8000)
                self.vector_store.heartbeat()
                console.print("[green]✅ Connected to ChromaDB[/green]")
            except Exception as e:
                console.print(f"[yellow]⚠️  ChromaDB not running[/yellow]")
                console.print("[dim]Start it with: reddress vector start chromadb[/dim]")
                raise
        
        elif self.vector_store_type == "qdrant":
            try:
                from qdrant_client import QdrantClient
                self.vector_store = QdrantClient(host="localhost", port=6333)
                console.print("[green]✅ Connected to Qdrant[/green]")
            except Exception as e:
                console.print(f"[yellow]⚠️  Qdrant not running[/yellow]")
                console.print("[dim]Start it with: reddress vector start qdrant[/dim]")
                raise
        
        elif self.vector_store_type == "milvus":
            try:
                from pymilvus import connections
                connections.connect(host="localhost", port="19530")
                console.print("[green]✅ Connected to Milvus[/green]")
            except Exception as e:
                console.print(f"[yellow]⚠️  Milvus not running[/yellow]")
                console.print("[dim]Start it with: reddress vector start milvus[/dim]")
                raise
        
        else:
            raise ValueError(f"Unsupported vector store: {self.vector_store_type}")
    
    def index_project(
        self, 
        path: str, 
        project_name: Optional[str] = None,
        recursive: bool = True,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        force: bool = False
    ) -> bool:
        """Index a project/folder into vector store with incremental updates"""
        
        try:
            # Generate project name if not provided
            if not project_name:
                project_name = Path(path).name
            
            console.print(f"\n[bold cyan]📚 Indexing Project: {project_name}[/bold cyan]\n")
            
            # Check if project already exists
            existing_project = self.db.get_project(project_name)
            
            if existing_project and not force:
                console.print("[cyan]🔍 Checking for new or modified files...[/cyan]\n")
                return self._incremental_index(
                    path, project_name, existing_project, 
                    recursive, chunk_size, chunk_overlap
                )
            
            # Full index (first time or forced)
            if force and existing_project:
                console.print("[yellow]⚠️  Force re-indexing all files...[/yellow]\n")
            
            # Step 1: Load documents
            console.print("[cyan]Step 1/4:[/cyan] Loading documents...")
            loader = DocumentLoader()
            documents = loader.load_directory(path, recursive=recursive)
            
            if not documents:
                console.print("[yellow]⚠️  No documents found to index[/yellow]")
                return False
            
            console.print(f"[green]✅ Loaded {len(documents)} files[/green]\n")
            
            # Calculate file hashes
            file_hashes = {}
            for doc in documents:
                file_path = doc['metadata']['source']
                file_hashes[file_path] = self.db.get_file_hash(file_path)
            
            # Step 2: Chunk documents
            console.print("[cyan]Step 2/4:[/cyan] Chunking documents...")
            chunker = SmartChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = chunker.chunk_documents(documents)
            console.print(f"[green]✅ Created {len(chunks)} chunks[/green]\n")
            
            # Step 3: Generate embeddings
            console.print("[cyan]Step 3/4:[/cyan] Generating embeddings...")
            embedding_generator = EmbeddingGenerator(model=self.embedding_model)
            
            texts = [chunk['content'] for chunk in chunks]
            embeddings = embedding_generator.generate_embeddings(texts)
            console.print()
            
            # Step 4: Store in vector database
            console.print("[cyan]Step 4/4:[/cyan] Storing in vector database...")
            self._store_embeddings(project_name, chunks, embeddings)
            console.print(f"[green]✅ Stored {len(embeddings)} embeddings[/green]\n")
            
            # Save project metadata
            project_data = {
                'name': project_name,
                'path': os.path.abspath(path),
                'vector_store': self.vector_store_type if not self.use_memory else 'memory',
                'embedding_model': self.embedding_model,
                'total_files': len(documents),
                'total_chunks': len(chunks),
                'chunk_size': chunk_size,
                'chunk_overlap': chunk_overlap,
                'recursive': recursive,
                'indexed_at': datetime.now().isoformat(),
                'embedding_tokens': embedding_generator.total_tokens,
                'embedding_cost': embedding_generator.calculate_cost(embedding_generator.total_tokens),
                'file_hashes': file_hashes
            }
            
            self.db.add_project(project_data)
            
            console.print(f"[bold green]🎉 Successfully indexed '{project_name}'![/bold green]")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Error during indexing: {str(e)}[/red]")
            import traceback
            traceback.print_exc()
            return False
    
    def _incremental_index(
        self,
        path: str,
        project_name: str,
        existing_project: Dict,
        recursive: bool,
        chunk_size: int,
        chunk_overlap: int
    ) -> bool:
        """Incrementally index only new or modified files"""
        
        try:
            # Get existing file hashes
            existing_hashes = existing_project.get('file_hashes', {})
            
            # Load all documents
            loader = DocumentLoader()
            all_documents = loader.load_directory(path, recursive=recursive)
            
            # Filter to only new or modified files
            new_or_modified = []
            unchanged_count = 0
            new_hashes = {}
            
            for doc in all_documents:
                file_path = doc['metadata']['source']
                current_hash = self.db.get_file_hash(file_path)
                new_hashes[file_path] = current_hash
                
                # Check if file is new or modified
                if file_path not in existing_hashes or existing_hashes[file_path] != current_hash:
                    new_or_modified.append(doc)
                else:
                    unchanged_count += 1
            
            # Check for deleted files
            deleted_files = set(existing_hashes.keys()) - set(new_hashes.keys())
            
            if not new_or_modified and not deleted_files:
                console.print(f"[green]✅ All {len(all_documents)} files are up to date![/green]")
                console.print("[dim]No changes detected. Index is current.[/dim]\n")
                return True
            
            console.print(f"[cyan]📊 Index Status:[/cyan]")
            console.print(f"  • Unchanged: {unchanged_count} files")
            console.print(f"  • New/Modified: {len(new_or_modified)} files")
            console.print(f"  • Deleted: {len(deleted_files)} files\n")
            
            if len(new_or_modified) == 0 and len(deleted_files) > 0:
                console.print("[yellow]⚠️  Only deletions detected. Consider full re-index with --force[/yellow]\n")
                return True
            
            # Chunk only new/modified documents
            console.print("[cyan]Chunking new/modified files...[/cyan]")
            chunker = SmartChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            new_chunks = chunker.chunk_documents(new_or_modified)
            console.print(f"[green]✅ Created {len(new_chunks)} new chunks[/green]\n")
            
            # Generate embeddings
            console.print("[cyan]Generating embeddings...[/cyan]")
            embedding_generator = EmbeddingGenerator(model=self.embedding_model)
            texts = [chunk['content'] for chunk in new_chunks]
            embeddings = embedding_generator.generate_embeddings(texts)
            console.print()
            
            # Append to existing vector store
            console.print("[cyan]Appending to vector database...[/cyan]")
            self._append_embeddings(project_name, new_chunks, embeddings)
            console.print(f"[green]✅ Added {len(embeddings)} new embeddings[/green]\n")
            
            # Update project metadata
            total_chunks = existing_project.get('total_chunks', 0) + len(new_chunks)
            total_cost = existing_project.get('embedding_cost', 0) + embedding_generator.calculate_cost(embedding_generator.total_tokens)
            
            project_data = {
                **existing_project,
                'total_files': len(all_documents),
                'total_chunks': total_chunks,
                'indexed_at': datetime.now().isoformat(),
                'embedding_tokens': existing_project.get('embedding_tokens', 0) + embedding_generator.total_tokens,
                'embedding_cost': total_cost,
                'file_hashes': new_hashes
            }
            
            self.db.add_project(project_data)
            
            console.print(f"[bold green]🎉 Successfully updated '{project_name}'![/bold green]")
            console.print(f"[dim]Total: {len(all_documents)} files, {total_chunks} chunks[/dim]\n")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Error during incremental indexing: {str(e)}[/red]")
            import traceback
            traceback.print_exc()
            return False
    
    def _store_embeddings(self, collection_name: str, chunks: List[Dict], embeddings: List[List[float]]):
        """Store embeddings in vector store or memory"""
        
        if self.use_memory:
            self._store_memory(collection_name, chunks, embeddings)
        elif self.vector_store_type == "chromadb":
            self._store_chromadb(collection_name, chunks, embeddings)
        elif self.vector_store_type == "qdrant":
            self._store_qdrant(collection_name, chunks, embeddings)
        elif self.vector_store_type == "milvus":
            self._store_milvus(collection_name, chunks, embeddings)
    
    def _append_embeddings(self, collection_name: str, chunks: List[Dict], embeddings: List[List[float]]):
        """Append new embeddings to existing collection"""
        
        if self.use_memory:
            # Append to memory
            if collection_name in self.memory_store:
                existing = self.memory_store[collection_name]
                existing['chunks'].extend(chunks)
                existing['embeddings'] = np.vstack([existing['embeddings'], np.array(embeddings)])
            else:
                self._store_memory(collection_name, chunks, embeddings)
        
        elif self.vector_store_type == "chromadb":
            # Get existing collection
            collection = self.vector_store.get_collection(name=collection_name)
            existing_count = collection.count()
            
            # Prepare data
            ids = [f"chunk_{existing_count + i}" for i in range(len(chunks))]
            documents = [chunk['content'] for chunk in chunks]
            metadatas = [chunk['metadata'] for chunk in chunks]
            
            # Append to collection
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
        
        elif self.vector_store_type == "qdrant":
            from qdrant_client.models import PointStruct
            
            # Get existing collection info
            collection_info = self.vector_store.get_collection(collection_name=collection_name)
            existing_count = collection_info.points_count
            
            # Prepare points
            points = [
                PointStruct(
                    id=existing_count + i,
                    vector=embeddings[i],
                    payload={
                        'content': chunks[i]['content'],
                        **chunks[i]['metadata']
                    }
                )
                for i in range(len(chunks))
            ]
            
            # Append points
            self.vector_store.upsert(
                collection_name=collection_name,
                points=points
            )
        
        elif self.vector_store_type == "milvus":
            from pymilvus import Collection
            collection = Collection(collection_name)
            
            # Prepare data
            data = [
                embeddings,
                [chunk['content'][:65535] for chunk in chunks],
                [chunk['metadata']['source'][:1000] for chunk in chunks]
            ]
            
            # Insert data
            collection.insert(data)
            collection.flush()
    
    def _store_memory(self, collection_name: str, chunks: List[Dict], embeddings: List[List[float]]):
        """Store in memory"""
        self.memory_store[collection_name] = {
            'chunks': chunks,
            'embeddings': np.array(embeddings)
        }
        console.print(f"[green]✅ Stored {len(chunks)} chunks in memory[/green]")
    
    def _store_chromadb(self, collection_name: str, chunks: List[Dict], embeddings: List[List[float]]):
        """Store in ChromaDB"""
        try:
            self.vector_store.delete_collection(name=collection_name)
        except:
            pass
        
        collection = self.vector_store.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        documents = [chunk['content'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            collection.add(
                ids=ids[i:i+batch_size],
                embeddings=embeddings[i:i+batch_size],
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size]
            )
    
    def _store_qdrant(self, collection_name: str, chunks: List[Dict], embeddings: List[List[float]]):
        """Store in Qdrant"""
        from qdrant_client.models import Distance, VectorParams, PointStruct
        
        try:
            self.vector_store.delete_collection(collection_name=collection_name)
        except:
            pass
        
        self.vector_store.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=len(embeddings[0]), distance=Distance.COSINE)
        )
        
        points = [
            PointStruct(
                id=i,
                vector=embeddings[i],
                payload={
                    'content': chunks[i]['content'],
                    **chunks[i]['metadata']
                }
            )
            for i in range(len(chunks))
        ]
        
        batch_size = 100
        for i in range(0, len(points), batch_size):
            self.vector_store.upsert(
                collection_name=collection_name,
                points=points[i:i+batch_size]
            )
    
    def _store_milvus(self, collection_name: str, chunks: List[Dict], embeddings: List[List[float]]):
        """Store in Milvus"""
        from pymilvus import CollectionSchema, FieldSchema, DataType, Collection, utility
        
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
        
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=len(embeddings[0])),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1000),
        ]
        
        schema = CollectionSchema(fields, description=f"RAG collection for {collection_name}")
        collection = Collection(collection_name, schema)
        
        data = [
            embeddings,
            [chunk['content'][:65535] for chunk in chunks],
            [chunk['metadata']['source'][:1000] for chunk in chunks]
        ]
        
        collection.insert(data)
        collection.flush()
        
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        collection.create_index("embeddings", index_params)
    
    def load_project(self, project_name: str):
        """Load a project for querying"""
        project_data = self.db.get_project(project_name)
        
        if not project_data:
            raise ValueError(f"Project '{project_name}' not found in database")
        
        # Debug info
        console.print(f"[dim]Loading project: {project_name}[/dim]")
        console.print(f"[dim]Vector store: {self.vector_store_type}[/dim]")
        
        if self.use_memory or project_data.get('vector_store') == 'memory':
            if project_name not in self.memory_store:
                console.print(f"[yellow]⚠️  Project '{project_name}' not in memory[/yellow]")
                console.print(f"[yellow]💡 Re-index with: reddress rag index {project_data.get('path', '.')} --name {project_name} --vector memory[/yellow]")
                raise ValueError(f"Project '{project_name}' not in memory")
            else:
                self.collection = project_name
                console.print(f"[green]✅ Loaded project from memory: {project_name}[/green]")
        else:
            if self.vector_store_type == "chromadb":
                try:
                    # List available collections for debugging
                    all_collections = self.vector_store.list_collections()
                    collection_names = [c.name for c in all_collections]
                    console.print(f"[dim]Available collections: {collection_names}[/dim]")
                    
                    if project_name not in collection_names:
                        console.print(f"[red]❌ Collection '{project_name}' not found in ChromaDB[/red]")
                        console.print(f"[yellow]💡 Re-index with: reddress rag update {project_name}[/yellow]")
                        console.print(f"[yellow]💡 Or force re-index: reddress rag index {project_data.get('path', '.')} --name {project_name} --force[/yellow]")
                        raise ValueError(f"Collection '{project_name}' does not exist in ChromaDB")
                    
                    self.collection = self.vector_store.get_collection(name=project_name)
                    console.print(f"[green]✅ Loaded ChromaDB collection: {project_name}[/green]")
                except Exception as e:
                    console.print(f"[red]❌ Failed to load collection '{project_name}': {str(e)}[/red]")
                    raise
            
            elif self.vector_store_type == "qdrant":
                self.collection = project_name
                console.print(f"[green]✅ Loaded Qdrant collection: {project_name}[/green]")
            
            elif self.vector_store_type == "milvus":
                from pymilvus import Collection
                self.collection = Collection(project_name)
                self.collection.load()
                console.print(f"[green]✅ Loaded Milvus collection: {project_name}[/green]")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for relevant chunks"""
        if not self.collection:
            raise ValueError("No project loaded. Use load_project() first.")
        
        embedding_generator = EmbeddingGenerator(model=self.embedding_model)
        query_embedding = embedding_generator.generate_embeddings([query])[0]
        
        if self.use_memory:
            return self._search_memory(query_embedding, top_k)
        
        try:
            if self.vector_store_type == "chromadb":
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    include=['documents', 'metadatas', 'distances']
                )
                
                return [
                    {
                        'content': results['documents'][0][i],
                        'file': results['metadatas'][0][i].get('source', 'unknown'),
                        'score': 1 - results['distances'][0][i],
                        'metadata': results['metadatas'][0][i]
                    }
                    for i in range(len(results['documents'][0]))
                ]
            
            elif self.vector_store_type == "qdrant":
                try:
                    results = self.vector_store.query_points(
                        collection_name=self.collection,
                        query=query_embedding,
                        limit=top_k,
                        with_payload=True
                    )
                    points = results.points
                except AttributeError:
                    try:
                        results = self.vector_store.search(
                            collection_name=self.collection,
                            query_vector=query_embedding,
                            limit=top_k,
                            with_payload=True
                        )
                        points = results
                    except:
                        from qdrant_client.http import models as rest
                        results = self.vector_store.search(
                            collection_name=self.collection,
                            query_vector=query_embedding,
                            limit=top_k
                        )
                        points = results
                
                return [
                    {
                        'content': point.payload.get('content', ''),
                        'file': point.payload.get('source', 'unknown'),
                        'score': point.score if hasattr(point, 'score') else 0.0,
                        'metadata': point.payload if hasattr(point, 'payload') else {}
                    }
                    for point in points
                ]
            
            elif self.vector_store_type == "milvus":
                search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
                results = self.collection.search(
                    data=[query_embedding],
                    anns_field="embeddings",
                    param=search_params,
                    limit=top_k,
                    output_fields=["content", "source"]
                )
                
                return [
                    {
                        'content': hit.entity.get('content', ''),
                        'file': hit.entity.get('source', 'unknown'),
                        'score': hit.score,
                        'metadata': {'source': hit.entity.get('source', 'unknown')}
                    }
                    for hit in results[0]
                ]
        
        except Exception as e:
            console.print(f"[red]❌ Search error: {str(e)}[/red]")
            import traceback
            traceback.print_exc()
            return []
    
    def _search_memory(self, query_embedding: List[float], top_k: int) -> List[Dict]:
        """Search in-memory storage"""
        collection_data = self.memory_store[self.collection]
        embeddings = collection_data['embeddings']
        chunks = collection_data['chunks']
        
        query_vec = np.array(query_embedding)
        
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        query_norm = query_vec / np.linalg.norm(query_vec)
        
        similarities = np.dot(embeddings_norm, query_norm)
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'content': chunks[idx]['content'],
                'file': chunks[idx]['metadata']['source'],
                'score': float(similarities[idx]),
                'metadata': chunks[idx]['metadata']
            })
        
        return results
    
    def query(self, query: str, top_k: int = 5, model: str = "gpt-4o-mini") -> Dict:
        """Query with RAG - search + generate answer (legacy method)"""
        
        results = self.search(query, top_k=top_k)
        
        if not results:
            return {
                'answer': "I couldn't find any relevant information in the codebase.",
                'sources': []
            }
        
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(f"[Source {i}: {result['file']}]\n{result['content']}\n")
        
        context = "\n---\n".join(context_parts)
        
        try:
            from .helpers import get_openai_key
            api_key = get_openai_key()
        except:
            try:
                from ..utils import get_openai_key
                api_key = get_openai_key()
            except:
                api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            return {
                'answer': "OpenAI API key not configured.",
                'sources': results
            }
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            
            system_prompt = """You are a helpful coding assistant. Answer questions about the codebase based on the provided context.

Rules:
- Only use information from the provided context
- Be concise but complete
- Include code snippets when relevant
- If you're unsure, say so
- Reference specific files when mentioning code"""

            user_prompt = f"""Context from codebase:

{context}

---

Question: {query}

Please answer based on the context above."""

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            # Fixed: Proper response handling
            if hasattr(response, 'choices') and len(response.choices) > 0:
                choice = response.choices[0]
                if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                    answer = choice.message.content
                else:
                    answer = str(choice)
            else:
                answer = "Unable to generate response"
            
            return {
                'answer': answer,
                'sources': [
                    {
                        'file': r['file'],
                        'score': r['score'],
                        'content': r['content'][:200] + "..."
                    }
                    for r in results
                ],
                'tokens': response.usage.total_tokens if hasattr(response, 'usage') else 0,
                'cost': self._calculate_llm_cost(model, response.usage.prompt_tokens if hasattr(response, 'usage') else 0, response.usage.completion_tokens if hasattr(response, 'usage') else 0)
            }
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                'answer': f"Error: {str(e)}",
                'sources': results
            }
    
    def query_with_prompt(self, query: str, top_k: int = 5, system_prompt: str = None, model: str = "gpt-4o-mini", show_spinner: bool = True) -> Dict:
        """Query with RAG using custom prompt template"""
        
        # Search (animation handled by CLI)
        results = self.search(query, top_k=top_k)
        
        if not results:
            return {
                'answer': "I couldn't find any relevant information in the codebase.",
                'sources': []
            }
        
        # Build context from results
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(f"[Source {i}: {result['file']}]\n{result['content']}\n")
        
        context = "\n---\n".join(context_parts)
        
        # Get API key
        try:
            from .helpers import get_openai_key
            api_key = get_openai_key()
        except:
            try:
                from ..utils import get_openai_key
                api_key = get_openai_key()
            except:
                api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            return {
                'answer': "OpenAI API key not configured.",
                'sources': results
            }
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            
            # Use custom system prompt or default
            if not system_prompt:
                system_prompt = """You are a helpful coding assistant. Answer questions about the codebase based on the provided context.

Rules:
- Only use information from the provided context
- Be concise but complete
- Include code snippets when relevant (use ```
- Use markdown formatting for better readability
- If you're unsure, say so
- Reference specific files when mentioning code"""

            user_prompt = f"""Context from codebase:

{context}

***

Question: {query}

Please answer based on the context above."""

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            # Fixed: Proper response handling
            if hasattr(response, 'choices') and len(response.choices) > 0:
                choice = response.choices[0]
                if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                    answer = choice.message.content
                else:
                    answer = str(choice)
            else:
                answer = "Unable to generate response"

            try:
                answer = response.choices[0].message.content
            except (AttributeError, IndexError, TypeError) as e:
                console.print(f"[red]Error extracting answer: {e}[/red]")
                answer = "Unable to extract response content"
            
            return {
                'answer': answer,
                'sources': [
                    {
                        'file': r['file'],
                        'score': r['score'],
                        'content': r['content'][:200] + "..."
                    }
                    for r in results
                ],
                'tokens': response.usage.total_tokens if hasattr(response, 'usage') else 0,
                'cost': self._calculate_llm_cost(model, response.usage.prompt_tokens if hasattr(response, 'usage') else 0, response.usage.completion_tokens if hasattr(response, 'usage') else 0)
            }
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                'answer': f"Error: {str(e)}",
                'sources': results
            }
    
    def _calculate_llm_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate LLM cost"""
        pricing = {
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "gpt-4o-mini": {"input": 0.150, "output": 0.600},
            "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        }
        
        if model not in pricing:
            model = "gpt-4o-mini"
        
        input_cost = (input_tokens / 1_000_000) * pricing[model]["input"]
        output_cost = (output_tokens / 1_000_000) * pricing[model]["output"]
        
        return input_cost + output_cost
    
    def delete_collection(self, collection_name: str):
        """Delete a collection"""
        try:
            if self.use_memory:
                if collection_name in self.memory_store:
                    del self.memory_store[collection_name]
                    console.print(f"[green]✅ Deleted from memory: {collection_name}[/green]")
            elif self.vector_store_type == "chromadb":
                self.vector_store.delete_collection(name=collection_name)
                console.print(f"[green]✅ Deleted collection: {collection_name}[/green]")
            elif self.vector_store_type == "qdrant":
                self.vector_store.delete_collection(collection_name=collection_name)
                console.print(f"[green]✅ Deleted collection: {collection_name}[/green]")
            elif self.vector_store_type == "milvus":
                from pymilvus import utility
                utility.drop_collection(collection_name)
                console.print(f"[green]✅ Deleted collection: {collection_name}[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠️  Could not delete collection: {str(e)}[/yellow]")
