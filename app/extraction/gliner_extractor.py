import os
from typing import List, Optional
from app.extraction.contracts import ExtractorContract, Chunk, ExtractionCandidate

# Module-level model cache
_model_cache = {}


class GLiNERExtractor(ExtractorContract):
    def __init__(self, model_name: Optional[str] = None):
        self._model_name = model_name or os.environ.get("GLINER_MODEL", "urchade/gliner_medium-v2.1")

    def _get_model(self):
        if self._model_name in _model_cache:
            return _model_cache[self._model_name]
        try:
            from gliner import GLiNER
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = GLiNER.from_pretrained(self._model_name).to(device)
            _model_cache[self._model_name] = model
            return model
        except Exception as exc:
            raise RuntimeError(f"GLiNER model load failed: {exc}") from exc

    def extract_terms(
        self,
        chunks: List[Chunk],
        labels: List[str],
        threshold: float = 0.5,
        language_hint: Optional[str] = None,
    ) -> List[ExtractionCandidate]:
        model = self._get_model()
        candidates: List[ExtractionCandidate] = []
        
        # Check if model is on CUDA to set an appropriate batch size
        is_cuda = next(model.parameters()).device.type == "cuda"
        batch_size = 64 if is_cuda else 8
        texts = [chunk.content for chunk in chunks]
        if not texts:
            return []

        try:
            # Avoid batching memory explosion on 6GB VRAM by running sequentially on GPU
            # GLiNER on GPU is very fast even for sequential inference
            entities_list = []
            for text in texts:
                ents = model.predict_entities(text, labels, threshold=threshold)
                entities_list.append(ents)
        except Exception as exc:
            raise RuntimeError(f"GLiNER inference failed: {exc}") from exc

        for chunk, entities in zip(chunks, entities_list):
            for entity in entities:
                start = int(entity.get("start", -1))
                end = int(entity.get("end", -1))
                text = str(entity.get("text", ""))
                label = str(entity.get("label", ""))
                confidence = float(entity.get("score", 0.0))

                if start < 0 or end <= start or not text or not label:
                    continue

                candidates.append(
                    ExtractionCandidate(
                        chunk_id=chunk.chunk_id,
                        text=text,
                        label=label,
                        start_offset=start,
                        end_offset=end,
                        confidence=confidence,
                    )
                )

        return candidates
