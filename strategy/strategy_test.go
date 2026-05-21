package strategy

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// TestStrategyModeConstants verifies all strategy modes are defined.
func TestStrategyModeConstants(t *testing.T) {
	modes := []string{
		ModeSkipped,
		ModeDirectAnswer,
		ModeVector,
		ModeHybrid,
		ModeHybridRerank,
		ModeStructured,
		ModeMultiQuery,
		ModeHyDE,
	}

	for _, mode := range modes {
		require.NotEmpty(t, mode)
	}
}

// TestStrategyNameConstants verifies all strategy names are defined.
func TestStrategyNameConstants(t *testing.T) {
	names := []string{
		NameNone,
		NameVector,
		NameHybrid,
		NameHybridRerank,
		NameMultiQuery,
		NameHyDE,
	}

	for _, name := range names {
		require.NotEmpty(t, name)
	}
}

// TestMessageStruct tests the Message data structure.
func TestMessageStruct(t *testing.T) {
	message := Message{
		Role:    "user",
		Content: "Hello",
	}

	require.Equal(t, "user", message.Role)
	require.Equal(t, "Hello", message.Content)
}

// TestMessageRoles tests different message roles.
func TestMessageRoles(t *testing.T) {
	roles := []string{"user", "assistant", "system", "tool"}

	for _, role := range roles {
		msg := Message{Role: role, Content: "test"}
		require.Equal(t, role, msg.Role)
	}
}

// TestRetrievalHintsStruct tests the RetrievalHints data structure.
func TestRetrievalHintsStruct(t *testing.T) {
	hints := RetrievalHints{
		TopK:      5,
		Threshold: 0.75,
		TypeSlugs: []string{"faq", "guide"},
		Tags:      []string{"config"},
	}

	require.Equal(t, 5, hints.TopK)
	require.Equal(t, 0.75, hints.Threshold)
	require.Equal(t, 2, len(hints.TypeSlugs))
	require.Equal(t, 1, len(hints.Tags))
}

// TestQueryStruct tests the Query data structure.
func TestQueryStruct(t *testing.T) {
	query := Query{
		TenantID:      "t1",
		TenantAgentID: "a1",
		UserText:      "what is X",
		SystemPrompt:  "You are helpful",
		RAGIntent:     "faq",
		History: []Message{
			{Role: "user", Content: "q1"},
			{Role: "assistant", Content: "a1"},
		},
	}

	require.Equal(t, "t1", query.TenantID)
	require.Equal(t, "a1", query.TenantAgentID)
	require.Equal(t, "what is X", query.UserText)
	require.Equal(t, 2, len(query.History))
}

// TestQueryWithHints tests Query with retrieval hints.
func TestQueryWithHints(t *testing.T) {
	query := Query{
		UserText: "test",
		Hints: RetrievalHints{
			TopK:      10,
			Threshold: 0.8,
		},
	}

	require.Equal(t, 10, query.Hints.TopK)
	require.Equal(t, 0.8, query.Hints.Threshold)
}

// TestQueryWithSourceIDs tests Query with source filtering.
func TestQueryWithSourceIDs(t *testing.T) {
	query := Query{
		UserText:  "test",
		SourceIDs: []string{"source-1", "source-2", "source-3"},
	}

	require.Equal(t, 3, len(query.SourceIDs))
}

// TestQueryWithPreAnalyzedFilters tests using pre-analyzed filters.
func TestQueryWithPreAnalyzedFilters(t *testing.T) {
	filters := &QueryFilters{
		TypeSlugs: []string{"faq"},
		Tags:      []string{"billing"},
	}

	query := Query{
		UserText:           "test",
		PreAnalyzedFilters: filters,
	}

	require.NotNil(t, query.PreAnalyzedFilters)
	require.Equal(t, 1, len(query.PreAnalyzedFilters.TypeSlugs))
}

// TestRetrievedChunkStruct tests the RetrievedChunk data structure.
func TestRetrievedChunkStruct(t *testing.T) {
	chunk := RetrievedChunk{
		ID:       "chunk-1",
		Name:     "Getting Started",
		TypeSlug: "guide",
		Summary:  "A guide to getting started",
		Content:  "Full content here",
		URL:      "https://example.com/guide",
		Tags:     []string{"intro", "setup"},
		Score:    0.95,
	}

	require.Equal(t, "chunk-1", chunk.ID)
	require.Equal(t, "Getting Started", chunk.Name)
	require.Equal(t, 0.95, chunk.Score)
	require.Equal(t, 2, len(chunk.Tags))
}

// TestRetrievedChunkWithParent tests chunk with parent reference.
func TestRetrievedChunkWithParent(t *testing.T) {
	parentID := "chunk-parent"
	parentChunk := &RetrievedChunk{
		ID:      "chunk-parent",
		Content: "Parent content",
	}

	chunk := RetrievedChunk{
		ID:          "chunk-1",
		Content:     "Child content",
		ParentID:    &parentID,
		ParentChunk: parentChunk,
	}

	require.NotNil(t, chunk.ParentID)
	require.Equal(t, "chunk-parent", *chunk.ParentID)
	require.NotNil(t, chunk.ParentChunk)
	require.Equal(t, "Parent content", chunk.ParentChunk.Content)
}

// TestRetrievedChunkAttributes tests chunk with attributes.
func TestRetrievedChunkAttributes(t *testing.T) {
	chunk := RetrievedChunk{
		ID: "chunk-1",
		Attributes: map[string]any{
			"version":    "1.0",
			"author":     "John",
			"difficulty": 3,
		},
	}

	require.NotNil(t, chunk.Attributes)
	require.Equal(t, "1.0", chunk.Attributes["version"])
	require.Equal(t, "John", chunk.Attributes["author"])
	require.Equal(t, 3, chunk.Attributes["difficulty"])
}

// TestRetrievalTraceStruct tests the RetrievalTrace data structure.
func TestRetrievalTraceStruct(t *testing.T) {
	now := time.Now()
	trace := RetrievalTrace{
		Strategy:       "vector",
		StartedAt:      now,
		LatencyMs:      250,
		Intent:         "faq",
		SemanticQuery:  "how to configure",
		LexicalQuery:   "configure",
		ChunksCount:    5,
		Reranked:       true,
		CacheHit:       "l1",
		InputTokens:    100,
		EmbeddingModel: "text-embedding-3-small",
		LLMInputTokens: 500,
		CostMicroCents: 1500,
	}

	require.Equal(t, "vector", trace.Strategy)
	require.Equal(t, int64(250), trace.LatencyMs)
	require.Equal(t, 5, trace.ChunksCount)
	require.True(t, trace.Reranked)
	require.Equal(t, "l1", trace.CacheHit)
}

// TestRetrievalTraceNotes tests trace with notes.
func TestRetrievalTraceNotes(t *testing.T) {
	trace := RetrievalTrace{
		Strategy: "hybrid",
		Notes: []string{
			"used multi-query expansion",
			"reranked with jina-reranker",
			"cache miss on embedding",
		},
	}

	require.Equal(t, 3, len(trace.Notes))
}

// TestResultStruct tests the Result data structure.
func TestResultStruct(t *testing.T) {
	chunks := []RetrievedChunk{
		{ID: "chunk-1", Score: 0.95},
		{ID: "chunk-2", Score: 0.87},
	}

	result := Result{
		Mode:         "vector",
		DirectAnswer: "",
		Chunks:       chunks,
		Trace: RetrievalTrace{
			Strategy: "vector",
			ChunksCount: 2,
		},
	}

	require.Equal(t, "vector", result.Mode)
	require.Empty(t, result.DirectAnswer)
	require.Equal(t, 2, len(result.Chunks))
	require.Equal(t, int64(0), result.Trace.LatencyMs) // Default value
}

// TestResultDirectAnswer tests result with direct answer.
func TestResultDirectAnswer(t *testing.T) {
	result := Result{
		Mode:         "direct_answer",
		DirectAnswer: "Yes, we support that",
		Chunks:       []RetrievedChunk{},
		Trace: RetrievalTrace{
			Strategy: "direct_answer",
		},
	}

	require.Equal(t, "direct_answer", result.Mode)
	require.Equal(t, "Yes, we support that", result.DirectAnswer)
	require.Empty(t, result.Chunks)
}

// TestQueryFiltersStruct tests the QueryFilters data structure.
func TestQueryFiltersStruct(t *testing.T) {
	filters := QueryFilters{
		Intent:        "specific",
		SemanticQuery: "how to set up X",
		KeywordQuery:  "setup X",
		TypeSlugs:     []string{"guide"},
		Tags:          []string{"setup"},
		Keywords:      []string{"X", "configuration"},
		Attributes: map[string]any{
			"complexity": "medium",
		},
	}

	require.Equal(t, "specific", filters.Intent)
	require.Equal(t, "how to set up X", filters.SemanticQuery)
	require.Equal(t, "setup X", filters.KeywordQuery)
}

// TestMultipleChunkScores tests result with varied chunk scores.
func TestMultipleChunkScores(t *testing.T) {
	result := Result{
		Mode: "hybrid",
		Chunks: []RetrievedChunk{
			{ID: "c1", Score: 0.99},
			{ID: "c2", Score: 0.85},
			{ID: "c3", Score: 0.72},
			{ID: "c4", Score: 0.68},
			{ID: "c5", Score: 0.61},
		},
		Trace: RetrievalTrace{
			Strategy:    "hybrid_rerank",
			ChunksCount: 5,
			Reranked:    true,
		},
	}

	require.Equal(t, 5, len(result.Chunks))
	require.True(t, result.Trace.Reranked)

	// Verify scores are in expected range
	for _, chunk := range result.Chunks {
		require.Greater(t, chunk.Score, 0.5)
		require.LessOrEqual(t, chunk.Score, 1.0)
	}
}

// TestStrategyInterfaceContract verifies Strategy interface expectations.
func TestStrategyInterfaceContract(t *testing.T) {
	// This is a compile-time check; interface methods must be implemented
	var _ Strategy = (*TestMockStrategy)(nil)
}

// TestMockStrategy implements Strategy for interface testing.
type TestMockStrategy struct{}

func (m *TestMockStrategy) Name() string                       { return "test" }
func (m *TestMockStrategy) Retrieve(context interface{}, q Query) (*Result, error) { return nil, nil }
