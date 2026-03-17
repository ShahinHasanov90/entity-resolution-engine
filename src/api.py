"""FastAPI application for the multilingual entity resolver service."""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Optional

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .resolver import CompanyResolver, MatchResult
from .similarity import SimilarityScorer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class ResolveRequest(BaseModel):
    """Request body for the /resolve endpoint."""
    company_name: str = Field(..., min_length=1, description="Company name to resolve")
    top_k: int = Field(default=5, ge=1, le=50, description="Max results to return")
    threshold: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Override match threshold"
    )


class MatchResponse(BaseModel):
    """Single match entry in the resolve response."""
    name: str
    matched_variant: str
    score: float
    fuzzy_score: float
    semantic_score: float
    method: str
    company_id: Optional[int] = None


class ResolveResponse(BaseModel):
    """Response body for the /resolve endpoint."""
    query: str
    results: list[MatchResponse]
    count: int
    elapsed_ms: float


class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""
    status: str
    companies_loaded: int
    variants_indexed: int
    cache_stats: dict


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

_resolver: Optional[CompanyResolver] = None


def _load_config() -> dict:
    """Load configuration from settings.yaml."""
    config_path = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup/shutdown lifecycle: load model and company data."""
    global _resolver

    config = _load_config()
    sim_cfg = config.get("similarity", {})
    resolver_cfg = config.get("resolver", {})
    cache_cfg = config.get("cache", {})
    api_cfg = config.get("api", {})

    scorer = SimilarityScorer(
        model_name=sim_cfg.get("model_name", "all-MiniLM-L6-v2"),
        fuzzy_weight=sim_cfg.get("fuzzy_weight", 0.4),
        semantic_weight=sim_cfg.get("semantic_weight", 0.6),
    )

    logger.info("Loading sentence-BERT model...")
    scorer.load_model()

    _resolver = CompanyResolver(
        scorer=scorer,
        threshold=resolver_cfg.get("match_threshold", 0.85),
        fuzzy_top_k=resolver_cfg.get("fuzzy_top_k", 10),
        cache_max_size=cache_cfg.get("max_size", 10000),
        cache_ttl=cache_cfg.get("ttl_seconds", 3600.0),
    )

    data_path = api_cfg.get("company_data_path", "data/sample_companies.json")
    project_root = Path(__file__).resolve().parent.parent
    full_path = project_root / data_path
    if full_path.exists():
        count = _resolver.load_companies(full_path)
        logger.info("Loaded %d companies from %s", count, full_path)
    else:
        logger.warning("Company data file not found: %s", full_path)

    yield

    # Shutdown cleanup
    if _resolver:
        _resolver.cache.clear()


app = FastAPI(
    title="Multilingual Entity Resolver",
    description="Resolve company names across AZ/RU/EN using fuzzy matching and semantic similarity.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/resolve", response_model=ResolveResponse)
async def resolve_company(request: ResolveRequest) -> ResolveResponse:
    """Resolve a company name against the known entity registry.

    Returns ranked matches with similarity scores.
    """
    if _resolver is None:
        raise HTTPException(status_code=503, detail="Resolver not initialized")

    start = time.perf_counter()
    results: list[MatchResult] = _resolver.resolve(
        query=request.company_name,
        top_k=request.top_k,
        threshold=request.threshold,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    return ResolveResponse(
        query=request.company_name,
        results=[
            MatchResponse(
                name=r.name,
                matched_variant=r.matched_variant,
                score=r.score,
                fuzzy_score=r.fuzzy_score,
                semantic_score=r.semantic_score,
                method=r.method,
                company_id=r.company_id,
            )
            for r in results
        ],
        count=len(results),
        elapsed_ms=round(elapsed_ms, 2),
    )


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Service health check with resolver status."""
    if _resolver is None:
        raise HTTPException(status_code=503, detail="Resolver not initialized")

    return HealthResponse(
        status="healthy",
        companies_loaded=_resolver.company_count,
        variants_indexed=_resolver.variant_count,
        cache_stats=_resolver.cache.stats(),
    )
