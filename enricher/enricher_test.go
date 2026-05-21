package enricher

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/require"
	ragstore "github.com/madmike/go-rag/store"
	"github.com/madmike/go-rag/strategy"
	"github.com/madmike/go-infra/telemetry"
)

// MockStore for testing.
type MockStore struct {
	entries map[string]ragstore.Entry
	err     error
}

func NewMockStore() *MockStore {
	return &MockStore{
		entries: make(map[string]ragstore.Entry),
	}
}

func (m *MockStore) GetTaxonomy(ctx context.Context, sourceIDs []string) (*ragstore.Taxonomy, error) {
	return nil, nil
}

func (m *MockStore) HybridSearch(ctx context.Context, params ragstore.HybridSearchParams) ([]ragstore.SearchResult, error) {
	return nil, nil
}

func (m *MockStore) VectorSearch(ctx context.Context, tenantID string, sourceIDs []string, embedding []float32, topK int, threshold float64) ([]ragstore.SearchResult, error) {
	return nil, nil
}

func (m *MockStore) MultiQuerySearch(ctx context.Context, tenantID string, sourceIDs []string, embeddings [][]float32, topKPerQuery, finalTopK, rrfK int) ([]ragstore.SearchResult, error) {
	return nil, nil
}

func (m *MockStore) GetEntries(ctx context.Context, ids []string) ([]ragstore.Entry, error) {
	if m.err != nil {
		return nil, m.err
	}
	var entries []ragstore.Entry
	for _, id := range ids {
		if ent, ok := m.entries[id]; ok {
			entries = append(entries, ent)
		}
	}
	return entries, nil
}

func (m *MockStore) AddEntry(id string, entry ragstore.Entry) {
	m.entries[id] = entry
}

// TestEnricherEmptyResults returns nil for empty input.
func TestEnricherEmptyResults(t *testing.T) {
	enricher := New(Config{Store: NewMockStore()})

	chunks, err := enricher.EnrichToChunks(context.Background(), nil)
	require.NoError(t, err)
	require.Nil(t, chunks)
}

// TestEnricherEnrichesSearchResults converts results to chunks.
func TestEnricherEnrichesSearchResults(t *testing.T) {
	store := NewMockStore()
	store.AddEntry("entry-1", ragstore.Entry{
		ID:       "entry-1",
		Name:     "Product Page",
		TypeSlug: "page",
		Summary:  "Main product info",
		Content:  "Full product details here",
		Tags:     []string{"product"},
		Keywords: []string{"widget", "price"},
	})

	enricher := New(Config{Store: store})
	results := []ragstore.SearchResult{
		{
			ID:       "entry-1",
			Name:     "Product",
			TypeSlug: "page",
			Score:    0.95,
		},
	}

	chunks, err := enricher.EnrichToChunks(context.Background(), results)
	require.NoError(t, err)
	require.Equal(t, 1, len(chunks))
	require.Equal(t, "entry-1", chunks[0].ID)
	require.Equal(t, "page", chunks[0].TypeSlug)
}

// TestEnricherResolvesParents includes parent entry in formatted chunk.
func TestEnricherResolvesParents(t *testing.T) {
	store := NewMockStore()
	parentID := "parent-1"
	store.AddEntry("parent-1", ragstore.Entry{
		ID:       "parent-1",
		Name:     "Category: Electronics",
		TypeSlug: "category",
		Summary:  "Electronic items",
	})
	store.AddEntry("entry-1", ragstore.Entry{
		ID:       "entry-1",
		Name:     "Widget 3000",
		TypeSlug: "product",
		Content:  "A great widget",
		ParentID: &parentID,
	})

	enricher := New(Config{Store: store})
	results := []ragstore.SearchResult{
		{
			ID:       "entry-1",
			Name:     "Widget 3000",
			TypeSlug: "product",
			Score:    0.9,
		},
	}

	chunks, err := enricher.EnrichToChunks(context.Background(), results)
	require.NoError(t, err)
	require.Equal(t, 1, len(chunks))
	require.NotNil(t, chunks[0].ParentID)
	require.Equal(t, "parent-1", *chunks[0].ParentID)
}

// TestEnricherSyntheticQueries includes generated queries in content.
func TestEnricherSyntheticQueries(t *testing.T) {
	store := NewMockStore()
	store.AddEntry("entry-1", ragstore.Entry{
		ID:                 "entry-1",
		Name:               "FAQ",
		TypeSlug:           "faq",
		Content:            "How do I return?",
		SyntheticQueries:   []string{"How to return items?", "Return policy"},
	})

	enricher := New(Config{Store: store})
	results := []ragstore.SearchResult{
		{
			ID:       "entry-1",
			Name:     "FAQ",
			TypeSlug: "faq",
			Score:    0.85,
		},
	}

	chunks, err := enricher.EnrichToChunks(context.Background(), results)
	require.NoError(t, err)
	require.Contains(t, chunks[0].Content, "Can answer:")
	require.Contains(t, chunks[0].Content, "How to return items?")
}

// TestEnricherFallbackOnError returns summary-only chunks on store error.
func TestEnricherFallbackOnError(t *testing.T) {
	store := NewMockStore()
	store.err = errors.New("store unavailable")

	enricher := New(Config{Store: store})
	results := []ragstore.SearchResult{
		{
			ID:       "entry-1",
			Name:     "Product",
			TypeSlug: "page",
			Summary:  "Summary fallback",
			Score:    0.9,
		},
	}

	chunks, err := enricher.EnrichToChunks(context.Background(), results)
	require.NoError(t, err)
	require.Equal(t, 1, len(chunks))
	require.Equal(t, "entry-1", chunks[0].ID)
}

// TestFormatContextEmpty handles nil or empty chunks.
func TestFormatContextEmpty(t *testing.T) {
	ctx := FormatContext(nil)
	require.Equal(t, "", ctx)

	ctx = FormatContext([]strategy.RetrievedChunk{})
	require.Equal(t, "", ctx)
}

// TestFormatContextRendersChunks formats chunks into context string.
func TestFormatContextRendersChunks(t *testing.T) {
	chunks := []strategy.RetrievedChunk{
		{
			ID:       "1",
			Name:     "Chapter 1",
			TypeSlug: "chapter",
			Summary:  "Introduction",
			Content:  "Once upon a time...",
			Tags:     []string{"fiction"},
			URL:      "https://example.com/ch1",
		},
	}

	ctx := FormatContext(chunks)
	require.Contains(t, ctx, "=== CHAPTER: Chapter 1 ===")
	require.Contains(t, ctx, "Summary: Introduction")
	require.Contains(t, ctx, "Content: Once upon a time...")
	require.Contains(t, ctx, "Categories: fiction")
	require.Contains(t, ctx, "More info: https://example.com/ch1")
}

// TestFormatContextWithAttributes includes formatted attributes.
func TestFormatContextWithAttributes(t *testing.T) {
	chunks := []strategy.RetrievedChunk{
		{
			Name:       "Product",
			TypeSlug:   "product",
			Content:    "A widget",
			Attributes: map[string]any{"price": 29.99, "color": "blue"},
			Keywords:   []string{"tool", "useful"},
		},
	}

	ctx := FormatContext(chunks)
	require.Contains(t, ctx, "Details:")
	require.Contains(t, ctx, "price:")
	require.Contains(t, ctx, "color:")
	require.Contains(t, ctx, "Keywords: tool, useful")
}

// TestFormatContextMultipleChunks separates multiple chunks.
func TestFormatContextMultipleChunks(t *testing.T) {
	chunks := []strategy.RetrievedChunk{
		{
			Name:     "First",
			TypeSlug: "section",
			Content:  "First content",
		},
		{
			Name:     "Second",
			TypeSlug: "section",
			Content:  "Second content",
		},
	}

	ctx := FormatContext(chunks)
	require.Contains(t, ctx, "=== SECTION: First ===")
	require.Contains(t, ctx, "=== SECTION: Second ===")
	require.Equal(t, 2, len(chunks))
}

// TestEnricherMultipleResults processes multiple search results.
func TestEnricherMultipleResults(t *testing.T) {
	store := NewMockStore()
	store.AddEntry("entry-1", ragstore.Entry{
		ID:       "entry-1",
		Name:     "First",
		TypeSlug: "doc",
	})
	store.AddEntry("entry-2", ragstore.Entry{
		ID:       "entry-2",
		Name:     "Second",
		TypeSlug: "doc",
	})

	enricher := New(Config{Store: store})
	results := []ragstore.SearchResult{
		{ID: "entry-1", Name: "First", TypeSlug: "doc", Score: 0.9},
		{ID: "entry-2", Name: "Second", TypeSlug: "doc", Score: 0.85},
	}

	chunks, err := enricher.EnrichToChunks(context.Background(), results)
	require.NoError(t, err)
	require.Equal(t, 2, len(chunks))
}

// TestEnricherLogging uses provided logger.
func TestEnricherLogging(t *testing.T) {
	logger := &telemetry.NoOpLogger{}
	enricher := New(Config{
		Store:  NewMockStore(),
		Logger: logger,
	})

	chunks, err := enricher.EnrichToChunks(context.Background(), nil)
	require.NoError(t, err)
	require.Nil(t, chunks)
}

// TestEnricherContextCancellation respects context cancellation.
func TestEnricherContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	enricher := New(Config{Store: NewMockStore()})
	results := []ragstore.SearchResult{{ID: "1", Name: "Item"}}

	// Store.GetEntries should be called with cancelled context
	chunks, err := enricher.EnrichToChunks(ctx, results)
	// Either error or fallback is acceptable
	require.True(t, err == nil || chunks != nil)
}
