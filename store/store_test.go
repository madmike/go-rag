package store

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/require"
)

// MockStore for testing.
type MockStore struct {
	taxonomy *Taxonomy
	results  []SearchResult
	entries  map[string]Entry
	err      error
}

func NewMockStore() *MockStore {
	return &MockStore{
		entries: make(map[string]Entry),
		results: []SearchResult{},
	}
}

func (m *MockStore) GetTaxonomy(ctx context.Context, sourceIDs []string) (*Taxonomy, error) {
	if m.err != nil {
		return nil, m.err
	}
	return m.taxonomy, nil
}

func (m *MockStore) HybridSearch(ctx context.Context, params HybridSearchParams) ([]SearchResult, error) {
	if m.err != nil {
		return nil, m.err
	}
	return m.results, nil
}

func (m *MockStore) VectorSearch(ctx context.Context, tenantID string, sourceIDs []string, embedding []float32, topK int, threshold float64) ([]SearchResult, error) {
	if m.err != nil {
		return nil, m.err
	}
	return m.results, nil
}

func (m *MockStore) MultiQuerySearch(ctx context.Context, tenantID string, sourceIDs []string, embeddings [][]float32, topKPerQuery, finalTopK, rrfK int) ([]SearchResult, error) {
	if m.err != nil {
		return nil, m.err
	}
	return m.results, nil
}

func (m *MockStore) GetEntries(ctx context.Context, ids []string) ([]Entry, error) {
	if m.err != nil {
		return nil, m.err
	}
	var entries []Entry
	for _, id := range ids {
		if ent, ok := m.entries[id]; ok {
			entries = append(entries, ent)
		}
	}
	return entries, nil
}

func (m *MockStore) AddEntry(entry Entry) {
	m.entries[entry.ID] = entry
}

func (m *MockStore) AddSearchResult(result SearchResult) {
	m.results = append(m.results, result)
}

func (m *MockStore) SetTaxonomy(tax *Taxonomy) {
	m.taxonomy = tax
}

func (m *MockStore) SetError(err error) {
	m.err = err
}

// TestStoreGetTaxonomy retrieves vocabulary for sources.
func TestStoreGetTaxonomy(t *testing.T) {
	store := NewMockStore()
	tax := &Taxonomy{
		TypeSlugs: []string{"article", "product"},
		Tags:      []string{"featured", "sale"},
		Keywords:  []string{"new", "limited"},
	}
	store.SetTaxonomy(tax)

	result, err := store.GetTaxonomy(context.Background(), []string{"source-1"})
	require.NoError(t, err)
	require.Equal(t, 2, len(result.TypeSlugs))
	require.Contains(t, result.TypeSlugs, "product")
}

// TestStoreHybridSearch returns results from vector + full-text.
func TestStoreHybridSearch(t *testing.T) {
	store := NewMockStore()
	store.AddSearchResult(SearchResult{
		ID:       "result-1",
		Name:     "Article",
		TypeSlug: "article",
		Score:    0.95,
	})

	results, err := store.HybridSearch(context.Background(), HybridSearchParams{
		TenantID:       "tenant-1",
		QueryText:      "search term",
		SourceIDs:      []string{"source-1"},
		MatchCount:     10,
	})

	require.NoError(t, err)
	require.Equal(t, 1, len(results))
	require.Equal(t, "result-1", results[0].ID)
}

// TestStoreVectorSearch performs vector similarity.
func TestStoreVectorSearch(t *testing.T) {
	store := NewMockStore()
	store.AddSearchResult(SearchResult{
		ID:       "vec-result",
		Name:     "Match",
		TypeSlug: "doc",
		Score:    0.87,
	})

	results, err := store.VectorSearch(
		context.Background(),
		"tenant-1",
		[]string{"source-1"},
		[]float32{0.1, 0.2, 0.3},
		10,
		0.5,
	)

	require.NoError(t, err)
	require.Equal(t, 1, len(results))
	require.Equal(t, "vec-result", results[0].ID)
}

// TestStoreMultiQuerySearch runs RRF aggregation.
func TestStoreMultiQuerySearch(t *testing.T) {
	store := NewMockStore()
	store.AddSearchResult(SearchResult{
		ID:   "multi-result",
		Name: "Aggregated",
		Score: 0.78,
	})

	embeddings := [][]float32{
		{0.1, 0.2, 0.3},
		{0.2, 0.3, 0.4},
	}

	results, err := store.MultiQuerySearch(
		context.Background(),
		"tenant-1",
		[]string{"source-1"},
		embeddings,
		5,
		10,
		60,
	)

	require.NoError(t, err)
	require.Equal(t, 1, len(results))
}

// TestStoreGetEntries fetches full entry rows.
func TestStoreGetEntries(t *testing.T) {
	store := NewMockStore()
	store.AddEntry(Entry{
		ID:       "entry-1",
		TenantID: "tenant-1",
		Name:     "Product",
		TypeSlug: "product",
		Content:  "Full details",
	})

	entries, err := store.GetEntries(context.Background(), []string{"entry-1"})
	require.NoError(t, err)
	require.Equal(t, 1, len(entries))
	require.Equal(t, "Product", entries[0].Name)
	require.Equal(t, "Full details", entries[0].Content)
}

// TestStoreGetEntriesPartial returns only matching IDs.
func TestStoreGetEntriesPartial(t *testing.T) {
	store := NewMockStore()
	store.AddEntry(Entry{ID: "entry-1", Name: "First"})
	store.AddEntry(Entry{ID: "entry-2", Name: "Second"})

	entries, err := store.GetEntries(context.Background(), []string{"entry-1", "entry-99"})
	require.NoError(t, err)
	require.Equal(t, 1, len(entries))
	require.Equal(t, "entry-1", entries[0].ID)
}

// TestStoreEntryWithParent includes parent link.
func TestStoreEntryWithParent(t *testing.T) {
	store := NewMockStore()
	parentID := "parent-1"
	store.AddEntry(Entry{
		ID:       "child-1",
		Name:     "Child",
		ParentID: &parentID,
	})

	entries, err := store.GetEntries(context.Background(), []string{"child-1"})
	require.NoError(t, err)
	require.NotNil(t, entries[0].ParentID)
	require.Equal(t, "parent-1", *entries[0].ParentID)
}

// TestStoreEntryWithAttributes includes metadata map.
func TestStoreEntryWithAttributes(t *testing.T) {
	store := NewMockStore()
	store.AddEntry(Entry{
		ID:         "entry-1",
		Name:       "Product",
		Attributes: map[string]any{"price": 29.99, "color": "blue"},
	})

	entries, err := store.GetEntries(context.Background(), []string{"entry-1"})
	require.NoError(t, err)
	require.Equal(t, 29.99, entries[0].Attributes["price"])
	require.Equal(t, "blue", entries[0].Attributes["color"])
}

// TestStoreSearchResultWithLocations includes location data.
func TestStoreSearchResultWithLocations(t *testing.T) {
	store := NewMockStore()
	store.AddSearchResult(SearchResult{
		ID:        "result-1",
		Name:      "Location Item",
		Locations: map[string]any{"city": "NYC"},
	})

	results, err := store.HybridSearch(context.Background(), HybridSearchParams{})
	require.NoError(t, err)
	require.NotNil(t, results[0].Locations)
}

// TestStoreTaxonomyWithAttributes includes attribute schema.
func TestStoreTaxonomyWithAttributes(t *testing.T) {
	store := NewMockStore()
	tax := &Taxonomy{
		TypeSlugs:     []string{"product"},
		AttributeKeys: []string{"price", "color"},
		AttributeSchema: map[string]AttributeInfo{
			"price": {
				Type: "numeric",
				Min:  floatPtr(0),
				Max:  floatPtr(1000),
			},
			"color": {
				Type:   "categorical",
				Values: []string{"red", "blue", "green"},
			},
		},
	}
	store.SetTaxonomy(tax)

	result, err := store.GetTaxonomy(context.Background(), []string{"source-1"})
	require.NoError(t, err)
	require.Equal(t, 2, len(result.AttributeSchema))
	require.Equal(t, "numeric", result.AttributeSchema["price"].Type)
}

// TestStoreFilteringParams handles complex search filters.
func TestStoreFilteringParams(t *testing.T) {
	store := NewMockStore()
	store.AddSearchResult(SearchResult{ID: "filtered", Name: "Result"})

	results, err := store.HybridSearch(context.Background(), HybridSearchParams{
		TenantID:        "tenant-1",
		FilterTypeSlugs: []string{"product"},
		FilterTags:      []string{"featured"},
		FilterAttributes: map[string]any{"price": ">50"},
		MatchCount:      20,
		VectorWeight:    0.7,
		KeywordWeight:   0.3,
	})

	require.NoError(t, err)
	require.Equal(t, 1, len(results))
}

// TestStoreErrorHandling propagates errors.
func TestStoreErrorHandling(t *testing.T) {
	store := NewMockStore()
	store.SetError(errors.New("database error"))

	results, err := store.HybridSearch(context.Background(), HybridSearchParams{})
	require.Error(t, err)
	require.Nil(t, results)

	entries, err := store.GetEntries(context.Background(), []string{"id"})
	require.Error(t, err)
	require.Nil(t, entries)
}

// TestStoreEntryFullContent includes synthetic queries.
func TestStoreEntryFullContent(t *testing.T) {
	store := NewMockStore()
	store.AddEntry(Entry{
		ID:               "entry-1",
		Name:             "FAQ",
		SyntheticQueries: []string{"How to use?", "Getting started"},
	})

	entries, err := store.GetEntries(context.Background(), []string{"entry-1"})
	require.NoError(t, err)
	require.Equal(t, 2, len(entries[0].SyntheticQueries))
	require.Contains(t, entries[0].SyntheticQueries, "Getting started")
}

// TestStoreMultipleEntries returns all matches.
func TestStoreMultipleEntries(t *testing.T) {
	store := NewMockStore()
	for i := 1; i <= 5; i++ {
		store.AddEntry(Entry{ID: "entry-" + string(rune(48+i))})
	}

	entries, err := store.GetEntries(context.Background(), []string{"entry-1", "entry-2", "entry-3"})
	require.NoError(t, err)
	require.Equal(t, 3, len(entries))
}

func floatPtr(f float64) *float64 {
	return &f
}
