package strategy

import (
	"context"
	"fmt"
	"time"

	providercore "github.com/madmike/go-ai-providers/core"
	"github.com/madmike/go-infra/telemetry"
	"github.com/madmike/go-rerank"
	ragstore "github.com/madmike/go-rag/store"
)

// HybridRerankStrategyConfig contains the configuration
type HybridRerankStrategyConfig struct {
	Store              ragstore.Store
	LLMProvider        providercore.LLMProvider
	EmbeddingProvider  providercore.EmbeddingProvider
	EmbeddingModel     string
	QueryAnalyzerModel string
	Reranker           rerank.Reranker // nil → falls back to score-based truncation
	MaxChunks          int
	PreRerankChunks    int
	VectorWeight       float64
	KeywordWeight      float64
	Threshold          float64
	Logger             telemetry.Logger
}

// HybridRerankStrategy performs hybrid search followed by cross-encoder/LLM reranking
type HybridRerankStrategy struct {
	config HybridRerankStrategyConfig
	hybrid *HybridStrategy // Reuse hybrid strategy logic for initial retrieval
}

// NewHybridRerankStrategy creates a new hybrid rerank RAG strategy
func NewHybridRerankStrategy(config HybridRerankStrategyConfig) *HybridRerankStrategy {
	hybridConfig := HybridStrategyConfig{
		Store:              config.Store,
		LLMProvider:        config.LLMProvider,
		EmbeddingProvider:  config.EmbeddingProvider,
		EmbeddingModel:     config.EmbeddingModel,
		QueryAnalyzerModel: config.QueryAnalyzerModel,
		MaxChunks:          config.PreRerankChunks,
		VectorWeight:       config.VectorWeight,
		KeywordWeight:      config.KeywordWeight,
		Threshold:          config.Threshold,
		Logger:             config.Logger,
	}

	return &HybridRerankStrategy{
		config: config,
		hybrid: NewHybridStrategy(hybridConfig),
	}
}

// Name returns the strategy name
func (s *HybridRerankStrategy) Name() string {
	return NameHybridRerank
}

// Retrieve executes the retrieval
func (s *HybridRerankStrategy) Retrieve(ctx context.Context, q Query) (*Result, error) {
	logger := s.config.Logger.WithModule(s.Name())
	logger.Info("HybridRerankStrategy started retrieving", telemetry.String("query", q.UserText))
	start := time.Now()

	// 1. Initial retrieval using Hybrid Strategy
	// Override hints for initial retrieval to get more chunks
	preRerankK := s.config.PreRerankChunks
	if preRerankK <= 0 {
		preRerankK = 20
	}

	q.Hints.TopK = preRerankK
	res, err := s.hybrid.Retrieve(ctx, q)
	if err != nil {
		return nil, fmt.Errorf("initial hybrid retrieval failed: %w", err)
	}

	if res.Mode == ModeDirectAnswer || len(res.Chunks) == 0 {
		return res, nil // Return as is
	}

	// 2. Reranking: use the configured cross-encoder reranker.
	// Falls back to score-based truncation when no reranker is configured.
	topK := s.config.MaxChunks
	if topK <= 0 {
		topK = 5
	}

	rerankStart := time.Now()
	if s.config.Reranker != nil {
		candidates := make([]rerank.Candidate, len(res.Chunks))
		for i, ch := range res.Chunks {
			content := ch.Content
			if ch.Summary != "" && len(ch.Summary) < len(content) {
				content = ch.Summary + "\n" + content
			}
			candidates[i] = rerank.Candidate{ID: ch.ID, Content: content, Score: ch.Score}
		}

		ranked, rerankErr := s.config.Reranker.Rerank(ctx, q.UserText, candidates, topK)
		rerankMs := time.Since(rerankStart).Milliseconds()
		if rerankErr != nil {
			logger.Warn("reranker failed, falling back to score truncation",
				telemetry.Err(rerankErr),
				telemetry.Int64("rerank_ms", rerankMs),
			)
			// Graceful degradation: keep original order, just truncate.
			if len(res.Chunks) > topK {
				res.Chunks = res.Chunks[:topK]
			}
		} else {
			logger.Info("rag_rerank_done",
				telemetry.Int("candidates", len(candidates)),
				telemetry.Int("top_k", topK),
				telemetry.Int64("rerank_ms", rerankMs),
			)
			// Rebuild chunks slice in reranked order.
			chunkByID := make(map[string]RetrievedChunk, len(res.Chunks))
			for _, ch := range res.Chunks {
				chunkByID[ch.ID] = ch
			}
			rerankedChunks := make([]RetrievedChunk, 0, len(ranked))
			for _, r := range ranked {
				if ch, ok := chunkByID[r.ID]; ok {
					ch.Score = r.Score
					rerankedChunks = append(rerankedChunks, ch)
				}
			}
			res.Chunks = rerankedChunks
		}
		res.Trace.Notes = append(res.Trace.Notes, fmt.Sprintf("rerank:%dms", rerankMs))
	} else {
		if len(res.Chunks) > topK {
			res.Chunks = res.Chunks[:topK]
		}
	}

	// Update trace
	res.Mode = ModeHybridRerank
	res.Trace.Strategy = s.Name()
	res.Trace.Reranked = true
	res.Trace.ChunksCount = len(res.Chunks)
	res.Trace.LatencyMs = time.Since(start).Milliseconds()

	return res, nil
}
