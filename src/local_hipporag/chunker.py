# chunker.py


from local_vectors.embedders import vector_preprocessing, LocalEmbedder


class Chunker(LocalEmbedder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def chunk_text(self, text: str):
        return vector_preprocessing(
            text, 
            overlap=self.overlap,
            model_config=self.model_metadata,
            tokenizer=self.tokenizer,
            truncate=False
        )