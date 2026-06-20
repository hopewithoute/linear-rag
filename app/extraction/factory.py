from app.extraction.contracts import ExtractorContract
from app.extraction.regex_extractor import RegexExtractor
from app.extraction.pipeline import TermExtractionPipeline
from app.settings import Settings, get_settings

# GLiNER general concept labels
GLINER_LABELS = [
    "concept", "topic", "technology", "tool", "framework",
    "method", "process", "principle", "metric",
    "organization", "person", "location", "event",
    "disease", "treatment", "symptom", "drug",
    "protein", "gene", "virus", "species",
]


def create_term_extractor(settings: Settings | None = None) -> ExtractorContract:
    resolved = settings or get_settings()
    extractors: list[ExtractorContract] = []

    # Always add regex for structured patterns (error codes, versions, etc.)
    extractors.append(RegexExtractor())

    if resolved.NER_BACKEND == "remote" and resolved.TERM_EXTRACTOR_BASE_URL:
        try:
            from app.extraction.remote_extractor import RemoteExtractor
            extractors.append(RemoteExtractor(url=resolved.TERM_EXTRACTOR_BASE_URL, token=resolved.TERM_EXTRACTOR_API_KEY or "dev-secret-key"))
        except Exception as e:
            print(f"RemoteExtractor failed to load: {e}")
    elif resolved.NER_BACKEND == "gliner":
        try:
            from app.extraction.gliner_extractor import GLiNERExtractor
            extractors.append(GLiNERExtractor(model_name=resolved.GLINER_MODEL))
        except Exception as e:
            print(f"GLiNER failed to load: {e}")
    else:
        # Add spaCy for generic NER extraction
        try:
            from app.extraction.spacy_extractor import SpacyExtractor
            extractors.append(SpacyExtractor())
        except Exception:
            pass

    return TermExtractionPipeline(extractors)


def get_gliner_labels() -> list[str]:
    """Return GLiNER labels from settings or fallback to defaults."""
    resolved = get_settings()
    if resolved.NER_LABELS:
        return [L.strip() for L in resolved.NER_LABELS.split(",") if L.strip()]
    return GLINER_LABELS
