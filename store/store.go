package store

import "context"

// Store is the persistence interface used by RAG strategies and enrichers.
// It is intentionally narrow: only the methods required by the RAG pipeline.
// Implementations live outside this library (e.g. supabase adapter in go-storage).
type Store interface {
	// GetTaxonomy returns the available filter vocabulary for the given sources
	// (type slugs, tags, keywords, attribute keys). Used by query analyzers.
	GetTaxonomy(ctx context.Context, sourceIDs []string) (*Taxonomy, error)

	// HybridSearch performs combined vector + full-text search.
	HybridSearch(ctx context.Context, params HybridSearchParams) ([]SearchResult, error)

	// VectorSearch performs pure vector similarity search.
	VectorSearch(ctx context.Context, tenantID string, sourceIDs []string, embedding []float32, topK int, threshold float64) ([]SearchResult, error)

	// MultiQuerySearch runs multiple embeddings through RRF aggregation.
	MultiQuerySearch(ctx context.Context, tenantID string, sourceIDs []string, embeddings [][]float32, topKPerQuery, finalTopK, rrfK int) ([]SearchResult, error)

	// GetEntries fetches full entry rows by IDs (used by enrichers for parent resolution).
	GetEntries(ctx context.Context, ids []string) ([]Entry, error)
}

// AttributeInfo describes a single attribute key's type and value constraints.
type AttributeInfo struct {
	Type   string   `json:"type"`             // "numeric", "categorical", "text"
	Min    *float64 `json:"min,omitempty"`
	Max    *float64 `json:"max,omitempty"`
	Values []string `json:"values,omitempty"`
}

// Taxonomy holds available filter vocabulary for a set of sources.
type Taxonomy struct {
	TypeSlugs         []string                 `json:"type_slugs"`
	Tags              []string                 `json:"tags"`
	Keywords          []string                 `json:"keywords"`
	AttributeKeys     []string                 `json:"attribute_keys"`
	TypeAttributeKeys map[string][]string      `json:"type_attribute_keys"`
	AttributeSchema   map[string]AttributeInfo `json:"attribute_schema"`
}

// Entry is a full knowledge entry row returned by GetEntries.
type Entry struct {
	ID               string         `json:"id"`
	TenantID         string         `json:"tenant_id"`
	SourceID         string         `json:"source_id"`
	ParentID         *string        `json:"parent_id,omitempty"`
	TypeSlug         string         `json:"type_slug"`
	Name             string         `json:"name"`
	Summary          string         `json:"summary"`
	Content          string         `json:"content"`
	Tags             []string       `json:"tags"`
	Keywords         []string       `json:"keywords"`
	SyntheticQueries []string       `json:"synthetic_queries"`
	Attributes       map[string]any `json:"attributes"`
	URL              string         `json:"url"`
}

// SearchResult is a single hit returned by any search method.
type SearchResult struct {
	ID         string         `json:"id"`
	Name       string         `json:"name"`
	Content    string         `json:"content"`
	Summary    string         `json:"summary"`
	URL        string         `json:"url"`
	Score      float64        `json:"score"`
	TypeSlug   string         `json:"type_slug"`
	Tags       []string       `json:"tags"`
	Keywords   []string       `json:"keywords"`
	Attributes map[string]any `json:"attributes"`
	Locations  any            `json:"locations,omitempty"`
	ParentID   *string        `json:"parent_id,omitempty"`
}

// HybridSearchParams parameterises a hybrid vector + full-text search call.
type HybridSearchParams struct {
	TenantID         string         `json:"tenant_id"`
	QueryEmbedding   []float32      `json:"query_embedding,omitempty"`
	QueryText        string         `json:"query_text,omitempty"`
	SourceIDs        []string       `json:"source_ids,omitempty"`
	FilterTypeSlugs  []string       `json:"filter_type_slugs,omitempty"`
	FilterTags       []string       `json:"filter_tags,omitempty"`
	FilterAttributes map[string]any `json:"filter_attributes,omitempty"`
	// MatchCount is the maximum number of results to return.
	MatchCount     int     `json:"match_count,omitempty"`
	VectorWeight   float64 `json:"vector_weight,omitempty"`
	KeywordWeight  float64 `json:"keyword_weight,omitempty"`
	MatchThreshold float64 `json:"match_threshold,omitempty"`
}
