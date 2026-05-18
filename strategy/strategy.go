package strategy

import (
	"context"
	"time"
)

// Mode names returned in Result.Mode.
const (
	ModeSkipped      = "skipped"
	ModeDirectAnswer = "direct_answer"
	ModeVector       = "vector"
	ModeHybrid       = "hybrid"
	ModeHybridRerank = "hybrid_rerank"
	ModeStructured   = "structured"
	ModeMultiQuery   = "multi_query"
	ModeHyDE         = "hyde"
)

// Strategy names accepted in agent config.
const (
	NameNone         = "none"
	NameVector       = "vector"
	NameHybrid       = "hybrid"
	NameHybridRerank = "hybrid_rerank"
	NameMultiQuery   = "multi_query"
	NameHyDE         = "hyde"
)

// Message represents a chat message used as conversational context.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// RetrievalHints lets the caller override default retrieval parameters.
type RetrievalHints struct {
	TopK      int      `json:"top_k,omitempty"`
	Threshold float64  `json:"threshold,omitempty"`
	TypeSlugs []string `json:"type_slugs,omitempty"` // classifier-suggested type filters
	Tags      []string `json:"tags,omitempty"`        // classifier-suggested tag filters
}

// Query is the input passed to a Strategy.
type Query struct {
	TenantID      string         `json:"tenant_id"`
	TenantAgentID string         `json:"tenant_agent_id"`
	SourceIDs     []string       `json:"source_ids,omitempty"`
	UserText      string         `json:"user_text"`
	History       []Message      `json:"history,omitempty"`
	Hints         RetrievalHints `json:"hints,omitempty"`
	// SystemPrompt is used by direct-answer gates so the gate can decide
	// using the same persona context as the downstream LLM call.
	SystemPrompt string `json:"system_prompt,omitempty"`
	// RAGIntent is the classified intent of the query (faq, product_search,
	// service_lookup, booking, troubleshooting, generic). Used to gate
	// expensive sub-steps like the direct-answer check.
	RAGIntent string `json:"rag_intent,omitempty"`
	// PreAnalyzedFilters carries query filters already extracted by the
	// classifier's merged call. When set, strategies skip their own
	// analyzeQuery LLM call and use these filters directly (after taxonomy
	// validation). Also implies the direct-answer gate should be skipped,
	// since the classifier has already decided RAG is needed.
	PreAnalyzedFilters *QueryFilters `json:"pre_analyzed_filters,omitempty"`
}

// RetrievedChunk is the unit of context passed to the LLM.
type RetrievedChunk struct {
	ID          string          `json:"id"`
	Name        string          `json:"name,omitempty"`
	TypeSlug    string          `json:"type_slug,omitempty"`
	Summary     string          `json:"summary,omitempty"`
	Content     string          `json:"content,omitempty"`
	URL         string          `json:"url,omitempty"`
	Tags        []string        `json:"tags,omitempty"`
	Keywords    []string        `json:"keywords,omitempty"`
	Attributes  map[string]any  `json:"attributes,omitempty"`
	ParentID    *string         `json:"parent_id,omitempty"`
	ParentChunk *RetrievedChunk `json:"parent_chunk,omitempty"` // resolved parent section (Small-to-Big)
	Score       float64         `json:"score"`
}

// RetrievalTrace captures everything needed to reason about why a strategy
// chose what it did. It is meant to be persisted to platform.agent_rag_runs
// and surfaced in the UI.
type RetrievalTrace struct {
	Strategy      string         `json:"strategy"`
	StartedAt     time.Time      `json:"started_at"`
	LatencyMs     int64          `json:"latency_ms"`
	Intent        string         `json:"intent,omitempty"`
	SemanticQuery string         `json:"semantic_query,omitempty"`
	LexicalQuery  string         `json:"lexical_query,omitempty"`
	Filters       map[string]any `json:"filters,omitempty"`
	ChunksCount    int            `json:"chunks_count"`
	Reranked       bool           `json:"reranked"`
	CacheHit       string         `json:"cache_hit,omitempty"` // "" | "l1" | "l2"
	InputTokens    int            `json:"input_tokens,omitempty"`
	EmbeddingModel string         `json:"embedding_model,omitempty"`
	LLMInputTokens int            `json:"llm_input_tokens,omitempty"`
	LLMOutputTokens int           `json:"llm_output_tokens,omitempty"`
	CostMicroCents int64          `json:"cost_micro_cents,omitempty"`
	Notes          []string       `json:"notes,omitempty"`
}

// Result is the output of a Strategy.
type Result struct {
	Mode         string           `json:"mode"`
	DirectAnswer string           `json:"direct_answer,omitempty"`
	Chunks       []RetrievedChunk `json:"chunks,omitempty"`
	Trace        RetrievalTrace   `json:"trace"`
}

// Strategy retrieves context for an agent turn.
type Strategy interface {
	Name() string
	Retrieve(ctx context.Context, q Query) (*Result, error)
}
