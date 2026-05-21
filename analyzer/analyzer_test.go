package analyzer

import (
	"context"
	"testing"

	providercore "github.com/madmike/go-ai-providers/core"
	"github.com/madmike/go-infra/telemetry"
	ragcache "github.com/madmike/go-rag/cache"
	ragstore "github.com/madmike/go-rag/store"
	"github.com/stretchr/testify/require"
)

// MockLLMProvider implements providercore.LLMProvider for testing.
type MockLLMProvider struct {
	response string
	err      error
}

func (m *MockLLMProvider) ChatCompletion(ctx context.Context, req providercore.ChatRequest) (*providercore.ChatResponse, error) {
	if m.err != nil {
		return nil, m.err
	}
	return &providercore.ChatResponse{
		Content: m.response,
		Usage: providercore.Usage{
			InputTokens:  100,
			OutputTokens: 50,
		},
	}, nil
}

// TestNormalizeDirectQuery tests query normalization for caching.
func TestNormalizeDirectQuery(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"Hello World", "hello world"},
		{"What's up?", "what s up"},
		{"CAN YOU HELP?", "can you help"},
		{"  spaces  around  ", "spaces around"},
		{"multi-line\nquery", "multi line query"},
		{"", ""},
		{"   ", ""},
	}

	for _, tt := range tests {
		result := NormalizeDirectQuery(tt.input)
		require.Equal(t, tt.expected, result)
	}
}

// TestDirectAnswerCacheKey tests cache key generation with system prompt hashing.
func TestDirectAnswerCacheKey(t *testing.T) {
	query := "what is your name"
	systemPrompt := "You are a helpful assistant"

	key1 := DirectAnswerCacheKey(query, systemPrompt)
	require.NotEmpty(t, key1)
	require.Contains(t, key1, "v1:")

	// Same query + prompt = same key
	key2 := DirectAnswerCacheKey(query, systemPrompt)
	require.Equal(t, key1, key2)

	// Different prompt = different key
	key3 := DirectAnswerCacheKey(query, "You are a different assistant")
	require.NotEqual(t, key1, key3)

	// Different query = different key
	key4 := DirectAnswerCacheKey("who are you", systemPrompt)
	require.NotEqual(t, key1, key4)
}

// TestShouldRunDirectGate tests the gate eligibility check.
func TestShouldRunDirectGate(t *testing.T) {
	tests := []struct {
		query       string
		maxWords    int
		maxChars    int
		shouldRun   bool
	}{
		// Short queries that should run
		{"hello", 10, 120, true},
		{"what is your name", 10, 120, true},
		{"hi there", 5, 100, true},

		// Queries exceeding word limit
		{"one two three four five six seven eight nine ten eleven", 10, 500, false},

		// Queries exceeding char limit
		{"x" + "y" * 150, 50, 120, false},

		// Empty query
		{"", 10, 120, false},
		{"   ", 10, 120, false},

		// Custom limits
		{"hello world", 1, 120, false},
		{"hello", 10, 3, false},

		// Default limits (0 uses defaults)
		{"hello", 0, 0, true},
	}

	for _, tt := range tests {
		result := shouldRunDirectGate(tt.query, tt.maxWords, tt.maxChars)
		require.Equal(t, tt.shouldRun, result, "query=%q, maxWords=%d, maxChars=%d", tt.query, tt.maxWords, tt.maxChars)
	}
}

// TestHasCyrillic tests Cyrillic script detection (Russian).
func TestHasCyrillic(t *testing.T) {
	tests := []struct {
		text      string
		expected  bool
	}{
		{"привет", true},
		{"Hello привет", true},
		{"Привет мир", true},
		{"hello world", false},
		{"123 456", false},
		{"", false},
		{"ёЁ", true},
	}

	for _, tt := range tests {
		result := hasCyrillic(tt.text)
		require.Equal(t, tt.expected, result, "hasCyrillic(%q)", tt.text)
	}
}

// TestQueryFiltersStruct tests the QueryFilters data structure.
func TestQueryFiltersStruct(t *testing.T) {
	filters := QueryFilters{
		Intent:        "specific",
		SemanticQuery: "how do I configure X",
		KeywordQuery:  "configure X",
		SearchText:    "how do I configure X",
		TypeSlugs:     []string{"faq", "guide"},
		Tags:          []string{"configuration", "setup"},
		Keywords:      []string{"X", "setup"},
		Attributes: map[string]any{
			"category": "configuration",
			"level":    "advanced",
		},
	}

	require.Equal(t, "specific", filters.Intent)
	require.Equal(t, 2, len(filters.TypeSlugs))
	require.Equal(t, 2, len(filters.Tags))
	require.NotNil(t, filters.Attributes)
}

// TestAnalyzerWithNoOpConfig creates analyzer with minimal config.
func TestAnalyzerWithNoOpConfig(t *testing.T) {
	config := Config{
		LLMProvider:        &MockLLMProvider{response: "{}"},
		QueryAnalyzerModel: "gpt-4",
		Logger:             &telemetry.NoOpLogger{},
	}

	analyzer := New(config)
	require.NotNil(t, analyzer)
}

// TestAnalyzerAnalyzeQueryWithNilTaxonomy tests analyzer with nil taxonomy.
func TestAnalyzerAnalyzeQueryWithNilTaxonomy(t *testing.T) {
	llm := &MockLLMProvider{
		response: `{"intent":"general","semantic_query":"test query","keyword_query":"test","type_slugs":[],"tags":[],"keywords":[],"attributes":{}}`,
	}

	config := Config{
		LLMProvider:        llm,
		QueryAnalyzerModel: "gpt-4",
		Logger:             &telemetry.NoOpLogger{},
	}

	analyzer := New(config)
	filters, semantic, lexical := analyzer.AnalyzeQuery(context.Background(), "test query", nil)

	require.NotNil(t, filters)
	require.NotEmpty(t, semantic)
	require.NotEmpty(t, lexical)
}

// TestAnalyzerAnalyzeQueryWithTaxonomy tests analyzer with provided taxonomy.
func TestAnalyzerAnalyzeQueryWithTaxonomy(t *testing.T) {
	llm := &MockLLMProvider{
		response: `{"intent":"specific","semantic_query":"how to configure","keyword_query":"configure","type_slugs":["faq"],"tags":[],"keywords":[],"attributes":{}}`,
	}

	config := Config{
		LLMProvider:        llm,
		QueryAnalyzerModel: "gpt-4",
		Logger:             &telemetry.NoOpLogger{},
	}

	analyzer := New(config)
	taxonomy := &ragstore.Taxonomy{
		TypeSlugs: []string{"faq", "guide"},
		Tags:      []string{"config"},
	}

	filters, semantic, lexical := analyzer.AnalyzeQuery(context.Background(), "how to configure", taxonomy)

	require.NotNil(t, filters)
	require.Equal(t, "specific", filters.Intent)
	require.NotEmpty(t, semantic)
}

// TestTryDirectAnswerNoCache tests direct answer without cache.
func TestTryDirectAnswerNoCache(t *testing.T) {
	llm := &MockLLMProvider{
		response: `{"can_answer_immediately":false,"direct_answer":""}`,
	}

	config := Config{
		LLMProvider:        llm,
		QueryAnalyzerModel: "gpt-4",
		DirectGateMaxWords: 10,
		DirectGateMaxChars: 100,
		Logger:             &telemetry.NoOpLogger{},
	}

	analyzer := New(config)
	result := analyzer.TryDirectAnswer(context.Background(), "hello")

	require.False(t, result.Hit) // LLM said it can't answer directly
}

// TestTryDirectAnswerLongQuery skips gate for long queries.
func TestTryDirectAnswerLongQuery(t *testing.T) {
	llm := &MockLLMProvider{
		response: `{"can_answer_immediately":true,"direct_answer":"yes"}`,
	}

	config := Config{
		LLMProvider:        llm,
		QueryAnalyzerModel: "gpt-4",
		DirectGateMaxWords: 5,
		DirectGateMaxChars: 50,
		Logger:             &telemetry.NoOpLogger{},
	}

	analyzer := New(config)
	longQuery := "this is a very long query that exceeds the character limit"
	result := analyzer.TryDirectAnswer(context.Background(), longQuery)

	// Gate should not run for long queries
	require.False(t, result.Hit)
}

// TestDirectAnswerResultStruct tests the DirectAnswerResult structure.
func TestDirectAnswerResultStruct(t *testing.T) {
	result := DirectAnswerResult{
		Answer: "yes",
		Hit:    true,
		CacheHit: ragcache.TierL1,
	}

	require.Equal(t, "yes", result.Answer)
	require.True(t, result.Hit)
}

// TestAnalyzerConfig tests the Config structure.
func TestAnalyzerConfig(t *testing.T) {
	config := Config{
		LLMProvider:        &MockLLMProvider{},
		QueryAnalyzerModel: "gpt-4",
		SystemPrompt:       "You are helpful",
		DirectCacheTTL:     3600,
		DirectGateMaxWords: 10,
		DirectGateMaxChars: 100,
		Logger:             &telemetry.NoOpLogger{},
	}

	require.NotNil(t, config.LLMProvider)
	require.Equal(t, "gpt-4", config.QueryAnalyzerModel)
	require.Equal(t, int64(3600), config.DirectCacheTTL)
}

// TestAnalyzerErrorFallback tests fallback behavior on LLM error.
func TestAnalyzerErrorFallback(t *testing.T) {
	llm := &MockLLMProvider{
		err: ErrAnalyzerFailed, // Error response
	}

	config := Config{
		LLMProvider:        llm,
		QueryAnalyzerModel: "gpt-4",
		Logger:             &telemetry.NoOpLogger{},
	}

	analyzer := New(config)
	filters, semantic, lexical := analyzer.AnalyzeQuery(context.Background(), "test", nil)

	// Should fall back to original query
	require.Equal(t, "general", filters.Intent)
	require.Equal(t, "test", semantic)
	require.Equal(t, "test", lexical)
}

var ErrAnalyzerFailed = "analyzer error"
