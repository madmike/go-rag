package strategy

import (
	"context"
	"fmt"
	"strings"
	"time"

	providercore "github.com/madmike/go-ai-providers/core"
	"github.com/madmike/go-infra/telemetry"
	ragstore "github.com/madmike/go-rag/store"
)

// HyDEStrategyConfig contains configuration for HyDE (Hypothetical Document Embeddings)
type HyDEStrategyConfig struct {
	Store             ragstore.Store
	LLMProvider       providercore.LLMProvider
	EmbeddingProvider providercore.EmbeddingProvider
	EmbeddingModel    string
	HyDEModel         string // typically a small model for hypothesis generation
	MaxChunks         int
	Threshold         float32
	Logger            telemetry.Logger
}

// HyDEStrategy implements Hypothetical Document Embeddings:
// Generate a hypothetical ideal answer, embed it, and use for retrieval.
// This improves recall for short queries where user vocabulary differs from corpus.
type HyDEStrategy struct {
	config HyDEStrategyConfig
}

// NewHyDEStrategy creates a new HyDE retrieval strategy
func NewHyDEStrategy(config HyDEStrategyConfig) *HyDEStrategy {
	return &HyDEStrategy{
		config: config,
	}
}

// Name returns the strategy name
func (s *HyDEStrategy) Name() string {
	return "hyde"
}

// Retrieve executes HyDE retrieval: generate hypothesis → embed → search
func (s *HyDEStrategy) Retrieve(ctx context.Context, q Query) (*Result, error) {
	logger := s.config.Logger.WithModule(s.Name())
	logger.Info("HyDEStrategy started", telemetry.String("query", q.UserText))
	start := time.Now()

	trace := RetrievalTrace{
		Strategy:      s.Name(),
		StartedAt:     start,
		SemanticQuery: q.UserText,
	}

	// Step 1: Generate hypothetical ideal answer
	hydeStart := time.Now()
	hypothesis, err := s.generateHypothesis(ctx, q.UserText)
	if err != nil {
		logger.Warn("Failed to generate hypothesis, falling back to original query", telemetry.Err(err))
		hypothesis = q.UserText // fallback: use original query
	}
	ragStepNote(&trace, "hyde_gen", time.Since(hydeStart).Milliseconds())

	// Step 2: Embed hypothesis
	embStart := time.Now()
	embResp, err := s.config.EmbeddingProvider.GenerateEmbedding(ctx, providercore.EmbeddingRequest{
		Text:  hypothesis,
		Model: s.config.EmbeddingModel,
	})
	if err != nil || embResp == nil {
		logger.Error("Failed to embed hypothesis", telemetry.Err(err))
		trace.LatencyMs = time.Since(start).Milliseconds()
		return nil, fmt.Errorf("embedding hypothesis failed: %w", err)
	}
	ragStepNote(&trace, "embed", time.Since(embStart).Milliseconds())

	trace.EmbeddingModel = embResp.Model
	if embResp.Usage != nil {
		trace.InputTokens = embResp.Usage.InputTokens
	}

	// Step 3: Vector search using hypothesis embedding
	searchStart := time.Now()
	topK := s.config.MaxChunks
	if q.Hints.TopK > 0 {
		topK = q.Hints.TopK
	}
	threshold := float64(s.config.Threshold)
	if q.Hints.Threshold > 0 {
		threshold = q.Hints.Threshold
	}

	results, err := s.config.Store.VectorSearch(ctx, q.TenantID, q.SourceIDs, embResp.Vector, topK, threshold)
	if err != nil {
		logger.Error("Vector search failed", telemetry.Err(err))
		trace.LatencyMs = time.Since(start).Milliseconds()
		return nil, fmt.Errorf("vector search failed: %w", err)
	}
	ragStepNote(&trace, "search", time.Since(searchStart).Milliseconds())

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
	trace.Notes = append(trace.Notes, fmt.Sprintf("hypothesis:%q", truncateStr(hypothesis, 80)))

	return &Result{
		Mode:   "hyde",
		Chunks: chunks,
		Trace:  trace,
	}, nil
}

// generateHypothesis calls the LLM to generate a hypothetical ideal answer.
// Input: user's question. Output: 2-3 sentence ideal answer demonstrating what a correct response contains.
func (s *HyDEStrategy) generateHypothesis(ctx context.Context, query string) (string, error) {
	prompt := fmt.Sprintf(`Given the following question, generate a hypothetical ideal answer that would directly address it.
The answer should be 2-3 sentences, factual, and demonstrate the key concepts the user is asking about.
Question: %s

Generate only the answer, no preamble:`, query)

	resp, err := s.config.LLMProvider.ChatCompletion(ctx, providercore.ChatRequest{
		Model: s.config.HyDEModel,
		Messages: []providercore.Message{
			{
				Role:    "user",
				Content: prompt,
			},
		},
	})
	if err != nil || resp == nil || resp.Content == "" {
		return "", fmt.Errorf("hypothesis generation failed: %w", err)
	}

	return strings.TrimSpace(resp.Content), nil
}

// truncateStr truncates a string for logging
func truncateStr(s string, maxLen int) string {
	if len(s) > maxLen {
		return s[:maxLen] + "..."
	}
	return s
}
