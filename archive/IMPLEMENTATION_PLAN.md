# Sentinel Router - Implementation Plan

**Integrating:** Router_llm + Valanyx + Lost in the Middle

**Timeline:** 12 weeks to MVP launch

---

## Week-by-Week Implementation Plan

### Week 1-2: Core Integration Foundation

#### Week 1: Architecture Setup

**Goal:** Create monorepo structure with all 3 components

**Tasks:**
- [x] Day 1-2: Repository structure
  ```
  sentinel-router/
  ├── core/                    # Shared utilities
  ├── router/                  # Router_llm (intent detection)
  ├── engine/                  # Valanyx (steering + verification)
  ├── optimizer/               # Position-aware RAG
  ├── api/                     # FastAPI unified interface
  ├── tests/                   # Integration tests
  └── docs/                    # Documentation
  ```

- [x] Day 3-4: Dependency management
  - Consolidate requirements.txt
  - Resolve version conflicts
  - Set up virtual environment

- [x] Day 5: CI/CD pipeline
  - GitHub Actions setup
  - Linting (black, flake8)
  - Type checking (mypy)
  - Test automation (pytest)

**Deliverable:** Clean monorepo with all components

---

#### Week 2: Integration Layer

**Goal:** Connect Router → Valanyx → Position Optimizer

**Tasks:**
- [x] Day 1-2: Intent → Domain mapping
  ```python
  # core/intent_mapping.py
  INTENT_TO_DOMAIN = {
      "coding": "technical",
      "finance": "finance",
      "regulatory": "compliance",
      "creative": "general",
      "translation": "language",
  }
  ```

- [x] Day 3-4: Position-aware context builder
  ```python
  # optimizer/context_builder.py
  class PositionAwareContextBuilder:
      def __init__(self, model_info):
          self.model_size = detect_model_size(model_info)
          self.has_recency_bias = self.model_size <= 4  # 4B params

      def build(self, docs, model):
          if self.has_recency_bias:
              return self._build_reverse(docs)
          return self._build_standard(docs)
  ```

- [x] Day 5: Integration tests
  - Test intent → domain mapping
  - Test position-aware ordering
  - Test end-to-end flow

**Deliverable:** Working integration between all 3 components

---

### Week 3-4: Feature Implementation

#### Week 3: Position-Aware Steering

**Goal:** Combine steering with position optimization

**Tasks:**
- [x] Day 1-2: Position-aware intensity calculator
  ```python
  # optimizer/position_utils.py
  def calculate_position_intensity(base, position, total):
      """
      Adjust steering intensity based on position.

      Early positions (weak): Increase intensity
      Late positions (strong): Decrease intensity
      """
      position_pct = position / total

      if position_pct < 0.15:      # Position 1-15
          return base * 1.5        # Boost weak positions
      elif position_pct < 0.50:    # Position 16-50
          return base * 1.2
      elif position_pct < 0.75:    # Position 51-75
          return base * 1.0
      else:                        # Position 76-100
          return base * 0.7        # Reduce for strong positions
  ```

- [x] Day 3-4: Position-aware layer targeting
  ```python
  # optimizer/layer_targeting.py
  def get_position_aware_zone(position, total, model_depth):
      """
      Map document position to optimal layer range.

      Early docs → Early layers (inject before lost)
      Late docs → Late layers (reinforce attention)
      """
      position_pct = position / total

      if position_pct < 0.20:
          return (0.40, 0.55)  # Early layers
      elif position_pct < 0.60:
          return (0.55, 0.70)  # Middle layers
      else:
          return (0.70, 0.85)  # Late layers
  ```

- [x] Day 5: Integration with Valanyx steering
  - Modify `_register_hook()` to accept position params
  - Update steering vector calculation
  - Test position-aware steering

**Deliverable:** Position-aware activation steering working

---

#### Week 4: Verification Pipeline

**Goal:** 3-tier verification integrated with position metadata

**Tasks:**
- [x] Day 1-2: Enhanced verification with position tracking
  ```python
  # engine/verification.py
  class PositionAwareVerifier:
      def verify(self, query, answer, docs, position_metadata):
          """
          3-tier verification with position awareness.

          Returns verdict + position analysis
          """
          # Tier 1: Math check
          tier1_result = self.check_math(answer, docs)

          # Tier 2: NLI check
          tier2_result = self.check_nli(answer, docs[position_metadata['best_position']])

          # Tier 3: LLM judge (if needed)
          if tier1_result == "UNCLEAR" and tier2_result == "UNCLEAR":
              tier3_result = self.llm_judge(query, answer, docs)

          # Return with position metadata
          return {
              "verdict": final_verdict,
              "tier": verification_tier,
              "position_used": position_metadata['best_position'],
              "position_accuracy_expected": position_metadata['expected_accuracy']
          }
  ```

- [x] Day 3-4: Position analytics tracking
  - Track position vs accuracy correlation
  - Store results in database
  - Build analytics queries

- [x] Day 5: Testing
  - Verification accuracy tests
  - Position tracking tests
  - End-to-end integration tests

**Deliverable:** Complete verification pipeline with position analytics

---

### Week 5-6: API & SDK Development

#### Week 5: REST API

**Goal:** Unified API exposing all functionality

**Tasks:**
- [x] Day 1-2: Core endpoints
  ```python
  # api/main.py
  from fastapi import FastAPI
  from sentinel_router import SentinelRouter

  app = FastAPI(title="Sentinel Router API")
  router = SentinelRouter()

  @app.post("/v1/query")
  async def query(request: QueryRequest):
      """
      Main query endpoint with full integration.

      Request:
      {
        "query": "What is HIPAA requirement for PII?",
        "context_docs": [...],  # Optional
        "compliance_profile": "HIPAA",  # Optional
        "enable_steering": true,
        "enable_position_aware": true
      }

      Response:
      {
        "response": "HIPAA requires...",
        "metadata": {
          "intent": "regulatory",
          "model": "ollama/gemma2",
          "documents_used": 3,
          "best_doc_position": 3,
          "position_gain": "+10%",
          "steering_applied": true,
          "verification": "TIER_2_NLI",
          "confidence": 0.93
        }
      }
      """
      result = await router.query(
          query=request.query,
          context_docs=request.context_docs,
          compliance_profile=request.compliance_profile,
          enable_steering=request.enable_steering,
          enable_position_aware=request.enable_position_aware
      )
      return result

  @app.post("/v1/recommend")
  async def recommend(request: RecommendRequest):
      """Recommend model without executing query."""
      return router.recommend_model(request.query)

  @app.get("/v1/analytics")
  async def analytics():
      """Get position analytics and usage stats."""
      return router.get_analytics()
  ```

- [x] Day 3-4: Authentication & rate limiting
  - API key management
  - Rate limiting per tier (free, pro, enterprise)
  - Usage tracking

- [x] Day 5: Documentation
  - OpenAPI schema
  - Postman collection
  - Example requests

**Deliverable:** Production-ready REST API

---

#### Week 6: Python SDK

**Goal:** Developer-friendly Python SDK

**Tasks:**
- [x] Day 1-3: SDK implementation
  ```python
  # sdk/sentinel_router.py
  from typing import List, Dict, Optional
  import httpx

  class SentinelRouter:
      """
      Sentinel Router SDK - Position-aware RAG with safety steering.

      Example:
          router = SentinelRouter(api_key="your-key")

          response = router.query(
              "What's the HIPAA requirement?",
              context_docs=retrieved_docs,
              compliance_profile="HIPAA"
          )

          print(response.text)
          print(f"Position gain: {response.position_gain}")
          print(f"Confidence: {response.confidence}")
      """

      def __init__(self, api_key: str, base_url: str = "http://localhost:8000"):
          self.api_key = api_key
          self.base_url = base_url
          self.client = httpx.AsyncClient()

      async def query(
          self,
          query: str,
          context_docs: Optional[List[Dict]] = None,
          compliance_profile: Optional[str] = None,
          enable_steering: bool = True,
          enable_position_aware: bool = True
      ) -> QueryResponse:
          """Execute query with full Sentinel Router pipeline."""
          response = await self.client.post(
              f"{self.base_url}/v1/query",
              json={
                  "query": query,
                  "context_docs": context_docs,
                  "compliance_profile": compliance_profile,
                  "enable_steering": enable_steering,
                  "enable_position_aware": enable_position_aware
              },
              headers={"X-API-Key": self.api_key}
          )
          return QueryResponse(**response.json())

      async def recommend_model(self, query: str) -> str:
          """Get model recommendation without executing query."""
          response = await self.client.post(
              f"{self.base_url}/v1/recommend",
              json={"query": query}
          )
          return response.json()["recommended_model"]

      def get_analytics(self) -> Dict:
          """Get position analytics and usage statistics."""
          response = self.client.get(f"{self.base_url}/v1/analytics")
          return response.json()
  ```

- [x] Day 4: Examples & tutorials
  - Basic usage examples
  - Healthcare compliance example
  - Financial analysis example
  - Code documentation example

- [x] Day 5: Testing & documentation
  - SDK unit tests
  - Integration tests
  - API reference docs
  - Jupyter notebook examples

**Deliverable:** Published Python SDK on PyPI

---

### Week 7-8: Pro/Enterprise Features

#### Week 7: Analytics Dashboard

**Goal:** Position analytics and usage tracking dashboard

**Tasks:**
- [x] Day 1-2: Data collection
  - Position vs accuracy tracking
  - Intent distribution
  - Model usage statistics
  - Verification tier distribution

- [x] Day 3-4: Dashboard UI
  ```
  Dashboard Views:
  ├── Overview
  │   ├── Total queries (last 30 days)
  │   ├── Average accuracy
  │   ├── Position gain achieved
  │   └── Cost savings vs GPT-4
  ├── Position Analytics
  │   ├── Accuracy by position (chart)
  │   ├── Position strategy distribution
  │   ├── Model-specific position effects
  │   └── A/B test results
  ├── Intent Analysis
  │   ├── Intent distribution (pie chart)
  │   ├── Intent accuracy by model
  │   └── Intent routing effectiveness
  └── Model Usage
      ├── Local vs Cloud ratio
      ├── Cost per model
      ├── Latency per model
      └── Verification statistics
  ```

- [x] Day 5: Export & reporting
  - CSV export
  - PDF reports
  - Email alerts (anomalies, errors)

**Deliverable:** Analytics dashboard deployed

---

#### Week 8: Custom Knowledge Bases

**Goal:** Allow customers to upload custom knowledge bases

**Tasks:**
- [x] Day 1-2: KB upload & processing
  ```python
  # api/knowledge_base.py
  @app.post("/v1/kb/upload")
  async def upload_kb(
      file: UploadFile,
      domain: str,
      compliance_profile: Optional[str] = None
  ):
      """
      Upload custom knowledge base.

      Accepts: JSON, CSV, or text files
      Processes:
        1. Parse documents
        2. Generate embeddings
        3. Associate with domain
        4. Link to compliance profile (if applicable)
      """
      # Parse file
      docs = parse_kb_file(file)

      # Generate embeddings
      embedder = get_embedder()
      embeddings = embedder.encode([doc['text'] for doc in docs])

      # Store in database
      kb_id = store_kb(docs, embeddings, domain, compliance_profile)

      return {"kb_id": kb_id, "documents": len(docs)}
  ```

- [x] Day 3-4: Custom compliance profiles
  - HIPAA profile template
  - SOC2 profile template
  - GDPR profile template
  - Custom profile builder UI

- [x] Day 5: Testing
  - KB upload tests
  - Retrieval accuracy tests
  - Compliance verification tests

**Deliverable:** Custom KB upload working

---

### Week 9-10: Deployment & Infrastructure

#### Week 9: Containerization

**Goal:** Docker containers for all components

**Tasks:**
- [x] Day 1-2: Dockerfiles
  ```dockerfile
  # Dockerfile.api
  FROM python:3.11-slim

  # Install dependencies
  COPY requirements.txt .
  RUN pip install --no-cache-dir -r requirements.txt

  # Copy application
  COPY . /app
  WORKDIR /app

  # Expose API port
  EXPOSE 8000

  # Run API server
  CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
  ```

  ```dockerfile
  # Dockerfile.ollama
  FROM ollama/ollama:latest

  # Pre-pull models
  RUN ollama pull llama3
  RUN ollama pull gemma2

  EXPOSE 11434
  ```

- [x] Day 3-4: Docker Compose
  ```yaml
  # docker-compose.yml
  version: '3.8'

  services:
    api:
      build:
        context: .
        dockerfile: Dockerfile.api
      ports:
        - "8000:8000"
      environment:
        - OLLAMA_BASE_URL=http://ollama:11434
        - DATABASE_URL=postgresql://postgres:password@db:5432/sentinel
      depends_on:
        - db
        - ollama

    ollama:
      build:
        dockerfile: Dockerfile.ollama
      ports:
        - "11434:11434"
      volumes:
        - ollama_data:/root/.ollama

    db:
      image: postgres:15
      environment:
        - POSTGRES_DB=sentinel
        - POSTGRES_PASSWORD=password
      volumes:
        - postgres_data:/var/lib/postgresql/data

    redis:
      image: redis:7
      ports:
        - "6379:6379"

  volumes:
    ollama_data:
    postgres_data:
  ```

- [x] Day 5: Testing
  - Docker build tests
  - Container networking tests
  - Volume persistence tests

**Deliverable:** Docker containers working

---

#### Week 10: Kubernetes Deployment

**Goal:** Production Kubernetes manifests

**Tasks:**
- [x] Day 1-2: K8s manifests
  ```yaml
  # k8s/deployment.yaml
  apiVersion: apps/v1
  kind: Deployment
  metadata:
    name: sentinel-router-api
  spec:
    replicas: 3
    selector:
      matchLabels:
        app: sentinel-router-api
    template:
      metadata:
        labels:
          app: sentinel-router-api
      spec:
        containers:
        - name: api
          image: sentinel-router/api:latest
          ports:
          - containerPort: 8000
          env:
          - name: OLLAMA_BASE_URL
            value: "http://ollama-service:11434"
          resources:
            requests:
              memory: "2Gi"
              cpu: "1000m"
            limits:
              memory: "4Gi"
              cpu: "2000m"
  ```

- [x] Day 3-4: Helm charts
  - Chart for API
  - Chart for Ollama
  - Chart for database
  - Chart for Redis

- [x] Day 5: Testing
  - Local K8s deployment (minikube)
  - Load testing
  - Scaling tests

**Deliverable:** Production-ready K8s deployment

---

### Week 11-12: Documentation & Launch

#### Week 11: Documentation

**Goal:** Comprehensive documentation

**Tasks:**
- [x] Day 1-2: Technical docs
  - Architecture overview
  - API reference
  - SDK reference
  - Deployment guide

- [x] Day 3-4: User guides
  - Quick start guide
  - Healthcare use case tutorial
  - Financial use case tutorial
  - Code documentation use case

- [x] Day 5: Video tutorials
  - Installation & setup (5 min)
  - First query walkthrough (10 min)
  - Custom KB upload (15 min)
  - Analytics dashboard tour (10 min)

**Deliverable:** Complete documentation site

---

#### Week 12: Launch Preparation

**Goal:** Public beta launch

**Tasks:**
- [x] Day 1-2: Marketing materials
  - Product website
  - Demo video
  - Blog post: "Introducing Sentinel Router"
  - Social media content

- [x] Day 3: Beta program
  - Select 50 beta testers
  - Onboarding emails
  - Support channel setup (Discord/Slack)

- [x] Day 4: Launch
  - Product Hunt submission
  - Hacker News post
  - Reddit r/MachineLearning post
  - Twitter announcement

- [x] Day 5: Post-launch
  - Monitor feedback
  - Fix critical bugs
  - Update documentation
  - Thank beta testers

**Deliverable:** Public beta launched! 🚀

---

## Resource Requirements

### Team

**Core Team (5 people):**
1. **Tech Lead** (full-time)
   - Architecture decisions
   - Code reviews
   - Integration oversight

2. **Backend Engineer** (full-time)
   - API development
   - Integration work
   - Database optimization

3. **ML Engineer** (full-time)
   - Model integration
   - Position optimization
   - Steering implementation

4. **Frontend Engineer** (part-time, weeks 7-11)
   - Analytics dashboard
   - Documentation site

5. **DevOps Engineer** (part-time, weeks 9-10)
   - Docker/K8s setup
   - CI/CD pipeline
   - Monitoring

**Extended Team:**
- Product Manager (advisor)
- Designer (weeks 7, 11)
- Technical Writer (week 11)

---

### Infrastructure Costs

**Development (Weeks 1-12):**
- AWS/GCP credits: $500/month
- Development machines: $0 (local)
- CI/CD: $0 (GitHub Actions free tier)
- **Total:** $1,500

**Beta Launch (Month 1):**
- Cloud hosting: $200/month
- Database: $100/month
- CDN: $50/month
- Monitoring: $50/month
- **Total:** $400/month

**Production (After launch):**
- Scales with usage
- Local deployment option reduces costs

---

## Success Metrics

### Week 1-4 (Integration)
- [ ] All components integrated and working
- [ ] Accuracy improvement validated: >+7%
- [ ] Latency acceptable: <3s p95
- [ ] All integration tests passing

### Week 5-8 (Features)
- [ ] API stable and documented
- [ ] SDK published on PyPI
- [ ] Analytics dashboard deployed
- [ ] Custom KB upload working

### Week 9-10 (Deployment)
- [ ] Docker containers working
- [ ] K8s deployment successful
- [ ] Load tests passing (100 RPS)
- [ ] Monitoring operational

### Week 11-12 (Launch)
- [ ] Documentation complete
- [ ] 50 beta users onboarded
- [ ] Product Hunt launch: >200 upvotes
- [ ] First paying customer acquired

---

## Risk Management

### Technical Risks

**Risk:** Position optimization doesn't work in production
- **Likelihood:** Low (validated with 630 experiments)
- **Impact:** High
- **Mitigation:** A/B testing, gradual rollout, fallback to standard

**Risk:** Integration complexity causes delays
- **Likelihood:** Medium
- **Impact:** Medium
- **Mitigation:** Weekly checkpoints, modular design, clear interfaces

**Risk:** Performance issues at scale
- **Likelihood:** Medium
- **Impact:** High
- **Mitigation:** Load testing, caching, horizontal scaling

---

### Business Risks

**Risk:** Market doesn't understand position awareness
- **Likelihood:** Medium
- **Impact:** Medium
- **Mitigation:** Clear ROI messaging, case studies, demos

**Risk:** Open source alternatives emerge
- **Likelihood:** Low (unique combination)
- **Impact:** High
- **Mitigation:** Fast iteration, strong community, enterprise features

**Risk:** Large model improvements reduce advantage
- **Likelihood:** Low-Medium
- **Impact:** Medium
- **Mitigation:** Focus on cost (local models), compliance, speed

---

## Next Steps

### Immediate (This week)
1. ✅ Review product vision document
2. ✅ Approve implementation plan
3. [ ] Set up monorepo
4. [ ] Kick off Week 1 tasks

### Short-term (Next month)
1. [ ] Complete Weeks 1-4 (integration)
2. [ ] Run integration tests
3. [ ] Validate accuracy improvements
4. [ ] Demo to stakeholders

### Medium-term (Months 2-3)
1. [ ] Complete Weeks 5-8 (features)
2. [ ] Launch private beta
3. [ ] Gather user feedback
4. [ ] Iterate based on feedback

### Long-term (Months 4-6)
1. [ ] Public launch
2. [ ] First enterprise customer
3. [ ] Series A fundraising
4. [ ] Team expansion

---

## Appendix: Code Organization

### Monorepo Structure

```
sentinel-router/
├── packages/
│   ├── core/                 # Shared utilities
│   │   ├── __init__.py
│   │   ├── intent_mapping.py
│   │   ├── model_detection.py
│   │   └── utils.py
│   │
│   ├── router/               # Router_llm integration
│   │   ├── __init__.py
│   │   ├── intent_detector.py
│   │   ├── semantic_router.py
│   │   └── model_selector.py
│   │
│   ├── engine/               # Valanyx integration
│   │   ├── __init__.py
│   │   ├── enforcement_engine.py
│   │   ├── steering.py
│   │   └── verification.py
│   │
│   ├── optimizer/            # Position-aware RAG
│   │   ├── __init__.py
│   │   ├── position_utils.py
│   │   ├── context_builder.py
│   │   └── analytics.py
│   │
│   ├── api/                  # FastAPI server
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── routes/
│   │   │   ├── query.py
│   │   │   ├── recommend.py
│   │   │   ├── analytics.py
│   │   │   └── kb.py
│   │   └── models/
│   │       ├── request.py
│   │       └── response.py
│   │
│   └── sdk/                  # Python SDK
│       ├── __init__.py
│       ├── client.py
│       └── models.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
├── docs/
│   ├── architecture.md
│   ├── api-reference.md
│   └── tutorials/
│
├── deployment/
│   ├── docker/
│   │   ├── Dockerfile.api
│   │   ├── Dockerfile.ollama
│   │   └── docker-compose.yml
│   ├── k8s/
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   └── ingress.yaml
│   └── terraform/
│
├── scripts/
│   ├── setup.sh
│   ├── test.sh
│   └── deploy.sh
│
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── deploy.yml
│
├── pyproject.toml
├── README.md
└── LICENSE
```

---

**Implementation Plan Status:** ✅ **READY TO EXECUTE**

**Estimated Timeline:** 12 weeks to MVP + Beta launch
**Estimated Cost:** $10K-$15K (team time + infrastructure)
**Success Probability:** High (all components validated independently)

---

**Next Action:** Begin Week 1 tasks (monorepo setup)

**Document Version:** 1.0
**Last Updated:** January 25, 2026
