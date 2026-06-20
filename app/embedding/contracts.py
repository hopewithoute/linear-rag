from typing import List, Union


class EmbeddingContract:
    def create_embeddings(self, texts: Union[str, List[str]]) -> List[List[float]]:
        raise NotImplementedError
