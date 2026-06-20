from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.api.auth import verify_api_key
from app.storage.db import get_db_session
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from app.storage.models import Chunk, Mention
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


class DeleteResponse(BaseModel):
    status: str
    document_id: str
    chunks_deactivated: int


@router.delete("/documents/{document_id}", response_model=DeleteResponse)
async def delete_document(
    document_id: int,
    _: None = Depends(verify_api_key),
    session: AsyncSession = Depends(get_db_session),
):
    try:
        stmt = update(Chunk).where(Chunk.document_id == document_id).values(active=False)
        result = await session.execute(stmt)
        count = result.rowcount

        if count == 0:
            raise HTTPException(status_code=404, detail="Document not found")

        await session.commit()

        return DeleteResponse(
            status="soft_deleted",
            document_id=str(document_id),
            chunks_deactivated=count,
        )

    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        logger.exception(f"Delete failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
