from typing import List, Dict
import re

class SmartChunker:
    """Smart text chunking with overlap for better context"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """Chunk documents intelligently"""
        all_chunks = []
        
        for doc in documents:
            chunks = self._chunk_by_type(doc)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def _chunk_by_type(self, doc: Dict) -> List[Dict]:
        """Chunk based on file type"""
        file_type = doc['metadata'].get('type', 'other')
        content = doc['content']
        metadata = doc['metadata']
        
        if file_type == 'code':
            return self._chunk_code(content, metadata)
        elif file_type == 'documentation':
            return self._chunk_markdown(content, metadata)
        else:
            return self._chunk_text(content, metadata)
    
    def _chunk_code(self, content: str, metadata: Dict) -> List[Dict]:
        """Chunk code by functions/classes"""
        chunks = []
        
        # Try to split by functions/classes
        # Python example
        if metadata['extension'] == '.py':
            pattern = r'((?:^|\n)(?:def|class)\s+\w+.*?(?=\n(?:def|class)\s+|\Z))'
            matches = re.findall(pattern, content, re.DOTALL | re.MULTILINE)
            
            if matches and len(matches) > 1:
                for i, match in enumerate(matches):
                    chunks.append({
                        'content': match.strip(),
                        'metadata': {
                            **metadata,
                            'chunk_id': i,
                            'chunk_type': 'function/class',
                            'total_chunks': len(matches)
                        }
                    })
                return chunks
        
        # Fallback to regular chunking
        return self._chunk_text(content, metadata)
    
    def _chunk_markdown(self, content: str, metadata: Dict) -> List[Dict]:
        """Chunk markdown by sections"""
        chunks = []
        
        # Split by headers
        sections = re.split(r'\n#{1,6}\s+', content)
        
        for i, section in enumerate(sections):
            if len(section.strip()) > 0:
                # If section too large, chunk it
                if len(section) > self.chunk_size:
                    sub_chunks = self._chunk_text(section, metadata)
                    chunks.extend(sub_chunks)
                else:
                    chunks.append({
                        'content': section.strip(),
                        'metadata': {
                            **metadata,
                            'chunk_id': i,
                            'chunk_type': 'section',
                            'total_chunks': len(sections)
                        }
                    })
        
        return chunks if chunks else self._chunk_text(content, metadata)
    
    def _chunk_text(self, content: str, metadata: Dict) -> List[Dict]:
        """Basic text chunking with overlap"""
        chunks = []
        
        # Try to split by paragraphs first
        paragraphs = content.split('\n\n')
        
        current_chunk = ""
        chunk_id = 0
        
        for para in paragraphs:
            # If adding this paragraph exceeds chunk size
            if len(current_chunk) + len(para) > self.chunk_size:
                if current_chunk:
                    chunks.append({
                        'content': current_chunk.strip(),
                        'metadata': {
                            **metadata,
                            'chunk_id': chunk_id,
                            'chunk_type': 'text'
                        }
                    })
                    chunk_id += 1
                    
                    # Add overlap
                    words = current_chunk.split()
                    overlap_words = words[-self.chunk_overlap // 5:] if len(words) > self.chunk_overlap // 5 else words
                    current_chunk = ' '.join(overlap_words) + '\n\n' + para
                else:
                    current_chunk = para
            else:
                current_chunk += '\n\n' + para if current_chunk else para
        
        # Add last chunk
        if current_chunk:
            chunks.append({
                'content': current_chunk.strip(),
                'metadata': {
                    **metadata,
                    'chunk_id': chunk_id,
                    'chunk_type': 'text'
                }
            })
        
        # Update total chunks
        for chunk in chunks:
            chunk['metadata']['total_chunks'] = len(chunks)
        
        return chunks
