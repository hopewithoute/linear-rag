from typing import Optional


def normalize_tenant_id(tenant_id: Optional[str]) -> str:
    return tenant_id or ""
