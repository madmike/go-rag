package strategy

import (
	"context"
	"fmt"
	"time"

	providercore "github.com/madmike/go-ai-providers/core"
	"github.com/madmike/go-infra/telemetry"
	ragstore "github.com/madmike/go-rag/store"
)

// VectorStrategyConfig contains the configuration for the vector RAG strategy
type VectorStrategyConfig struct {
	Store             ragstore.Store
	EmbeddingProvider providercore.EmbeddingProvider
	EmbeddingModel    string
	MaxChunks         int
	Threshold         float32
	FallbackContent   string
	Logger            telemetry.Logger
}

// VectorStrategy performs pure semantic vector search on entries
type VectorStrategy struct {
	config VectorStrategyConfig
}

// NewVectorStrategy creates a new vector RAG strategy
func NewVectorStrategy(config VectorStrategyConfig) *VectorStrategy {
	return &VectorStrategy{
		config: config,
	}
}

// Name returns the strategy name
func (s *VectorStrategy) Name() string {
	return NameVector
}

// Retrieve executes the vector retrieval
func (s *VectorStrategy) Retrieve(ctx context.Context, q Query) (*Result, error) {
	logger := s.config.Logger.WithModule(s.Name())
	logger.Info("VectorStrategy started retrieving", telemetry.String("query", q.UserText))
	start := time.Now()

	trace := RetrievalTrace{
		Strategy:      s.Name(),
		StartedAt:     start,
		SemanticQuery: q.UserText,
	}

	// Generate Embedding
	embResp, err := s.config.EmbeddingProvider.GenerateEmbedding(ctx, providercore.EmbeddingRequest{
		Text:  q.UserText,
		Model: s.config.EmbeddingModel,
	})
	if err != nil || embResp == nil {
		logger.Error("Failed to generate embedding", telemetry.Err(err))
		trace.LatencyMs = time.Since(start).Milliseconds()
		return nil, fmt.Errorf("failed to generate embedding: %w", err)
	}

	trace.EmbeddingModel = embResp.Model
	if embResp.Usage != nil {
		trace.InputTokens = embResp.Usage.InputTokens
	} else {
		// Heuristic: 4 chars per token
		trace.InputTokens = len(q.UserText) / 4
		if trace.InputTokens == 0 && len(q.UserText) > 0 {
			trace.InputTokens = 1
		}
	}

	topK := s.config.MaxChunks
	if q.Hints.TopK > 0 {
		topK = q.Hints.TopK
	}
	threshold := float64(s.config.Threshold)
	if q.Hints.Threshold > 0 {
		threshold = q.Hints.Threshold
	}

	// Execute Vector Search
	results, err := s.config.Store.VectorSearch(ctx, q.TenantID, q.SourceIDs, embResp.Vector, topK, threshold)
	if err != nil {
		logger.Error("Vector search failed", telemetry.Err(err))
		trace.LatencyMs = time.Since(start).Milliseconds()
		return nil, fmt.Errorf("vector search failed: %w", err)
	}

	// Format results
	chunks := make([]RetrievedChunk, len(results))
	for i, r := range results {
		chunks[i] = RetrievedChunk{
			ID:         r.ID,
			Name:       r.Name,
			TypeSlug:   r.TypeSlug,
			Summary:    r.Summary,
			Content:    r.Content,
			URL:        r.URL,
			Tags:       r.Tags,
			Keywords:   r.Keywords,
			Attributes: r.Attributes,
			ParentID:   r.ParentID,
			Score:      float64(r.Score),
		}
	}

	chunks = resolveParents(ctx, s.config.Store, chunks)

	trace.ChunksCount = len(chunks)
	trace.LatencyMs = time.Since(start).Milliseconds()

	return &Result{
		Mode:   ModeVector,
		Chunks: chunks,
		Trace:  trace,
	}, nil
}
