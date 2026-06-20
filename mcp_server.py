"""LinearRag-RAG MCP Server.
Provides MCP tools for AI agents to interact with LinearRag-RAG.
"""
import httpx
import json
import os
from typing import Any, Dict, List, Optional

FRUGAL_RAG_URL = os.environ.get("FRUGAL_RAG_URL", "http://localhost:8002")
FRUGAL_RAG_KEY = os.environ.get("FRUGAL_RAG_KEY", "test-secret-key")

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {FRUGAL_RAG_KEY}",
}


def _request(method: str, path: str, data: Optional[Dict] = None) -> Dict[str, Any]:
    """Make a request to LinearRag-RAG API."""
    url = f"{FRUGAL_RAG_URL}{path}"
    try:
        with httpx.Client(timeout=30.0, headers=HEADERS) as client:
            if method == "GET":
                resp = client.get(url)
            else:
                resp = client.post(url, json=data)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP {e.response.status_code}: {e.response.text}"}
    except Exception as e:
        return {"error": str(e)}


async def handle_ingest(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handle linear_rag_rag_ingest tool call."""
    text = args.get("text", "")
    document_id = args.get("document_id")
    metadata = args.get("metadata", {})

    if not text:
        return {"error": "text is required"}

    return _request("POST", "/ingest/document", {
        "text": text,
        "document_id": document_id,
        "metadata": metadata,
    })


async def handle_retrieve(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handle linear_rag_rag_retrieve tool call."""
    query = args.get("query", "")
    top_k = args.get("top_k", 5)
    mode = args.get("mode", "hybrid_ppr")

    if not query:
        return {"error": "query is required"}

    result = _request("POST", "/retrieve", {
        "query": query,
        "top_k": top_k,
        "mode": mode,
    })

    # Format response for LLM consumption
    if "contexts" in result:
        formatted = []
        for ctx in result["contexts"]:
            formatted.append({
                "text": ctx.get("text", ""),
                "score": ctx.get("score", 0),
                "source": ctx.get("document_id", ""),
                "evidence": ctx.get("evidence_path", []),
            })
        result["formatted_contexts"] = formatted

    return result


async def handle_compile(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handle linear_rag_rag_compile tool call."""
    mode = args.get("mode", "full")
    return _request("POST", "/graph/compile", {"mode": mode})


async def handle_snapshot(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handle linear_rag_rag_snapshot tool call."""
    return _request("GET", "/graph/snapshot/active")


# MCP Tools Definition
TOOLS = [
    {
        "name": "linear_rag_rag_ingest",
        "description": "Ingest a document into LinearRag-RAG for later retrieval. The document will be chunked, entities extracted, and indexed.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The document text to ingest",
                },
                "document_id": {
                    "type": "string",
                    "description": "Optional document ID (auto-generated if not provided)",
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional metadata (type, source, etc.)",
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "linear_rag_rag_retrieve",
        "description": "Retrieve relevant contexts using LinearRag-RAG. Supports BM25, hybrid RRF, and hybrid PPR modes for multi-hop retrieval.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default: 5)",
                    "default": 5,
                },
                "mode": {
                    "type": "string",
                    "enum": ["bm25_only", "hybrid", "hybrid_ppr"],
                    "description": "Retrieval mode (default: hybrid_ppr)",
                    "default": "hybrid_ppr",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "linear_rag_rag_compile",
        "description": "Compile the graph snapshot. Run after ingesting documents to update the graph for PPR retrieval.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["full", "incremental", "compact"],
                    "description": "Compile mode (default: full)",
                    "default": "full",
                },
            },
        },
    },
    {
        "name": "linear_rag_rag_snapshot",
        "description": "Get information about the active graph snapshot.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]


# MCP Server Implementation
async def handle_tool_call(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Route tool call to appropriate handler."""
    handlers = {
        "linear_rag_rag_ingest": handle_ingest,
        "linear_rag_rag_retrieve": handle_retrieve,
        "linear_rag_rag_compile": handle_compile,
        "linear_rag_rag_snapshot": handle_snapshot,
    }

    handler = handlers.get(tool_name)
    if not handler:
        return {"error": f"Unknown tool: {tool_name}"}

    return await handler(args)


def get_tools() -> List[Dict[str, Any]]:
    """Return list of available MCP tools."""
    return TOOLS
