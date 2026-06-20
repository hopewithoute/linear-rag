import pytest
from app.extraction.normalizer import normalize_entity


def test_normalize_basic():
    assert normalize_entity("  Hello World  ") == "hello world"


def test_normalize_unicode():
    assert normalize_entity("München") == "münchen"


def test_normalize_code_preserved():
    assert normalize_entity("ERR_AUTH_403") == "ERR_AUTH_403"
    assert normalize_entity("OOM") == "OOM"


def test_normalize_punctuation_stripped():
    assert normalize_entity("payment-service!") == "payment-service"
    assert normalize_entity("(error)") == "error"


def test_normalize_whitespace():
    assert normalize_entity("hello   world") == "hello world"
