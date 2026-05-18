package strategy

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	providercore "github.com/madmike/go-ai-providers/core"
	"github.com/madmike/go-infra/jsonutil"
	"github.com/madmike/go-infra/telemetry"
	ragstore "github.com/madmike/go-rag/store"
)

// MultiQueryStrategyConfig contains the configuration for the multi-query RAG strategy
type MultiQueryStrategyConfig struct {
	Store              ragstore.Store
	LLMProvider        providercore.LLMProvider
	EmbeddingProvider  providercore.EmbeddingProvider
	EmbeddingModel     string
	QueryAnalyzerModel string
	MaxChunks          int
	Logger             telemetry.Logger
}

// MultiQueryStrategy generates multiple variations of the query, fetches vectors for each, and aggregates via RRF
type MultiQueryStrategy struct {
	config MultiQueryStrategyConfig
}

// NewMultiQueryStrategy creates a new multi-query RAG strategy
func NewMultiQueryStrategy(config MultiQueryStrategyConfig) *MultiQueryStrategy {
	return &MultiQueryStrategy{
		config: config,
	}
}

// Name returns the strategy name
func (s *MultiQueryStrategy) Name() string {
	return NameMultiQuery
}

// Retrieve executes the multi-query retrieval
func (s *MultiQueryStrategy) Retrieve(ctx context.Context, q Query) (*Result, error) {
	logger := s.config.Logger.WithModule(s.Name())
	logger.Info("MultiQueryStrategy started retrieving", telemetry.String("query", q.UserText))
	start := time.Now()

	trace := RetrievalTrace{
		Strategy:      s.Name(),
		StartedAt:     start,
		SemanticQuery: q.UserText,
	}

	// 1. Generate query variations using LLM
	prompt := fmt.Sprintf("Generate 3 alternative phrasings of this search query to improve retrieval. Return a JSON array of strings including the original.\n\nQuery: %s", q.UserText)

	temp := 0.7
	maxTokens := 200
	resp, err := s.config.LLMProvider.ChatCompletion(ctx, providercore.ChatRequest{
		Model:       s.config.QueryAnalyzerModel,
		Temperature: &temp,
		MaxTokens:   &maxTokens,
		JSONMode:    true,
		Messages: []providercore.Message{
			{Role: "user", Content: prompt},
		},
	})

	var queries []string
	if err == nil {
		if resp.Usage != nil {
			trace.LLMInputTokens += resp.Usage.InputTokens
			trace.LLMOutputTokens += resp.Usage.OutputTokens
		}
		respText := jsonutil.ExtractJSON(jsonutil.SanitizeJSON(resp.Content))
		_ = json.Unmarshal([]byte(respText), &queries)
	}

	// Fallback if LLM generation failed or returned bad format
	if len(queries) == 0 {
		queries = []string{q.UserText}
	}

	trace.Notes = append(trace.Notes, fmt.Sprintf("Generated %d queries", len(queries)))

	// 2. Generate embeddings for all queries
	var embeddings [][]float32
	for _, queryVar := range queries {
		embResp, err := s.config.EmbeddingProvider.GenerateEmbedding(ctx, providercore.EmbeddingRequest{
			Text:  queryVar,
			Model: s.config.EmbeddingModel,
		})
		if err == nil && embResp != nil {
			embeddings = append(embeddings, embResp.Vector)
			trace.EmbeddingModel = embResp.Model
			if embResp.Usage != nil {
				trace.InputTokens += embResp.Usage.InputTokens
			} else {
				trace.InputTokens += len(queryVar) / 4
			}
		}
	}

	if len(embeddings) == 0 {
		logger.Error("Failed to generate any embeddings")
		trace.LatencyMs = time.Since(start).Milliseconds()
		return nil, fmt.Errorf("failed to generate embeddings")
	}

	topK := s.config.MaxChunks
	if q.Hints.TopK > 0 {
		topK = q.Hints.TopK
	}

	// 3. Execute Multi-Query Search
	results, err := s.config.Store.MultiQuerySearch(
		ctx,
		q.TenantID,
		q.SourceIDs,
		embeddings,
		10,   // top_k_per_query
		topK, // final_top_k
		60,   // rrf_k
	)

	if err != nil {
		logger.Error("Multi-query search failed", telemetry.Err(err))
		trace.LatencyMs = time.Since(start).Milliseconds()
		return nil, fmt.Errorf("multi-query search failed: %w", err)
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
		Mode:   ModeMultiQuery,
		Chunks: chunks,
		Trace:  trace,
	}, nil
}
