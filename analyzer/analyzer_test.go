package analyzer

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	providercore "github.com/madmike/go-ai-providers/core"
	"github.com/madmike/go-infra/telemetry"
	ragcache "github.com/madmike/go-rag/cache"
	ragstore "github.com/madmike/go-rag/store"
	"github.com/stretchr/testify/require"
)

type mockLLMProvider struct {
	response string
	err      error
}

func (m *mockLLMProvider) Name() string                    { return "mock-llm" }
func (m *mockLLMProvider) Type() providercore.ProviderType { return providercore.ProviderTypeOpenAI }
func (m *mockLLMProvider) Initialize(ctx context.Context, cfg providercore.ProviderConfig) error {
	return nil
}
func (m *mockLLMProvider) Close() error                          { return nil }
func (m *mockLLMProvider) HealthCheck(ctx context.Context) error { return nil }
func (m *mockLLMProvider) Capabilities() []providercore.Capability {
	return []providercore.Capability{providercore.CapabilityLLM}
}
func (m *mockLLMProvider) SupportsCapability(cap providercore.Capability) bool {
	return cap == providercore.CapabilityLLM
}
func (m *mockLLMProvider) ChatCompletion(ctx context.Context, req providercore.ChatRequest) (*providercore.ChatResponse, error) {
	if m.err != nil {
		return nil, m.err
	}
	return &providercore.ChatResponse{
		Content: m.response,
		Usage: &providercore.Usage{
			InputTokens:  10,
			OutputTokens: 5,
			TotalTokens:  15,
		},
	}, nil
}
func (m *mockLLMProvider) StreamChatCompletion(ctx context.Context, req providercore.ChatRequest) (providercore.ChatStream, error) {
	return nil, errors.New("not implemented")
}

func TestNormalizeDirectQuery(t *testing.T) {
	require.Equal(t, "hello world", NormalizeDirectQuery("  Hello, World!  "))
	require.Equal(t, "what s up", NormalizeDirectQuery("What's up?"))
	require.Equal(t, "", NormalizeDirectQuery("   "))
}

func TestDirectAnswerCacheKeyStableAndPromptSensitive(t *testing.T) {
	k1 := DirectAnswerCacheKey("hello", "prompt-a")
	k2 := DirectAnswerCacheKey("hello", "prompt-a")
	k3 := DirectAnswerCacheKey("hello", "prompt-b")

	require.Equal(t, k1, k2)
	require.NotEqual(t, k1, k3)
	require.True(t, strings.HasPrefix(k1, "v1:"))
}

func TestAnalyzeQueryFallbackOnProviderError(t *testing.T) {
	a := New(Config{
		LLMProvider:        &mockLLMProvider{err: errors.New("boom")},
		QueryAnalyzerModel: "gpt-test",
		Logger:             &telemetry.NoOpLogger{},
	})

	filters, semantic, lexical := a.AnalyzeQuery(context.Background(), "test query", nil)
	require.Equal(t, "general", filters.Intent)
	require.Equal(t, "test query", semantic)
	require.Equal(t, "test query", lexical)
}

func TestAnalyzeQueryParsesStructuredResponse(t *testing.T) {
	a := New(Config{
		LLMProvider: &mockLLMProvider{
			response: `{"intent":"specific","semantic_query":"how to configure x","keyword_query":"configure x","type_slugs":["faq"],"tags":["setup"],"attributes":{"tier":"pro"}}`,
		},
		QueryAnalyzerModel: "gpt-test",
		Logger:             &telemetry.NoOpLogger{},
	})

	taxonomy := &ragstore.Taxonomy{
		TypeSlugs: []string{"faq", "guide"},
		Tags:      []string{"setup"},
	}

	filters, semantic, lexical := a.AnalyzeQuery(context.Background(), "how to configure x", taxonomy)
	require.Equal(t, "specific", filters.Intent)
	require.Equal(t, "how to configure x", semantic)
	require.Equal(t, "configure x", lexical)
	require.Equal(t, []string{"faq"}, filters.TypeSlugs)
}

func TestTryDirectAnswerCachesResult(t *testing.T) {
	l1 := ragcache.NewInMemoryDirectAnswerCache(100)
	tiered := ragcache.NewTieredDirectAnswerCache(l1, nil, time.Minute)

	a := New(Config{
		LLMProvider: &mockLLMProvider{
			response: `{"can_answer_immediately":true,"direct_answer":"Hello there!"}`,
		},
		QueryAnalyzerModel: "gpt-test",
		DirectAnswerCache:  tiered,
		DirectCacheTTL:     time.Minute,
		DirectGateMaxWords: 10,
		DirectGateMaxChars: 120,
		Logger:             &telemetry.NoOpLogger{},
	})

	first := a.TryDirectAnswer(context.Background(), "hello")
	require.True(t, first.Hit)
	require.Equal(t, "Hello there!", first.Answer)

	// Flip provider response to ensure the second hit comes from cache.
	a.cfg.LLMProvider = &mockLLMProvider{
		response: `{"can_answer_immediately":true,"direct_answer":"DIFFERENT"}`,
	}
	second := a.TryDirectAnswer(context.Background(), "hello")
	require.True(t, second.Hit)
	require.Equal(t, "Hello there!", second.Answer)
	require.Equal(t, ragcache.TierL1, second.CacheHit)
}

func TestTryDirectAnswerSkipsLongQueries(t *testing.T) {
	a := New(Config{
		LLMProvider:        &mockLLMProvider{response: `{"can_answer_immediately":true,"direct_answer":"x"}`},
		QueryAnalyzerModel: "gpt-test",
		DirectGateMaxWords: 2,
		DirectGateMaxChars: 12,
		Logger:             &telemetry.NoOpLogger{},
	})

	result := a.TryDirectAnswer(context.Background(), "this query is definitely too long")
	require.False(t, result.Hit)
	require.Empty(t, result.Answer)
}
