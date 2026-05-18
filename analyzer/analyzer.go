package analyzer

import (
	"context"
	"crypto/sha1"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"time"

	providercore "github.com/madmike/go-ai-providers/core"
	"github.com/madmike/go-infra/jsonutil"
	"github.com/madmike/go-infra/telemetry"
	ragcache "github.com/madmike/go-rag/cache"
	ragstore "github.com/madmike/go-rag/store"
)

// QueryFilters holds extracted filters from query analysis.
type QueryFilters struct {
	Intent        string         `json:"intent"`
	SemanticQuery string         `json:"semantic_query,omitempty"`
	KeywordQuery  string         `json:"keyword_query,omitempty"`
	SearchText    string         `json:"search_text,omitempty"`
	TypeSlugs     []string       `json:"type_slugs,omitempty"`
	Tags          []string       `json:"tags,omitempty"`
	Keywords      []string       `json:"keywords,omitempty"`
	Attributes    map[string]any `json:"attributes,omitempty"`
}

// Config configures the query analyzer and the direct-answer gate.
type Config struct {
	LLMProvider        providercore.LLMProvider
	QueryAnalyzerModel string
	SystemPrompt       string

	DirectAnswerCache  ragcache.DirectAnswerCache
	DirectCacheTTL     time.Duration
	DirectGateMaxWords int
	DirectGateMaxChars int

	Logger telemetry.Logger
}

// Analyzer builds structured query plans for hybrid retrieval. It also offers
// a direct-answer gate that short-circuits the pipeline for trivial queries.
type Analyzer struct {
	cfg Config
}

func New(cfg Config) *Analyzer { return &Analyzer{cfg: cfg} }

// AnalyzeQuery returns extracted filters and the (semantic, lexical) query
// pair to feed into hybrid retrieval. On error it returns sensible fallbacks
// using the raw query text.
func (a *Analyzer) AnalyzeQuery(ctx context.Context, query string, taxonomy *ragstore.Taxonomy) (*QueryFilters, string, string) {
	if taxonomy == nil {
		taxonomy = &ragstore.Taxonomy{}
	}

	// Build compact taxonomy block
	var taxLines []string
	if len(taxonomy.TypeSlugs) > 0 {
		taxLines = append(taxLines, "types: "+strings.Join(taxonomy.TypeSlugs, ", "))
	}
	if len(taxonomy.Tags) > 0 {
		taxLines = append(taxLines, "tags: "+strings.Join(taxonomy.Tags, ", "))
	}
	if len(taxonomy.AttributeKeys) > 0 {
		taxLines = append(taxLines, "attribute_keys: "+strings.Join(taxonomy.AttributeKeys, ", "))
	}
	taxonomyBlock := strings.Join(taxLines, "\n")

	prompt := fmt.Sprintf(`Plan a hybrid RAG query. Write semantic_query and keyword_query in the same language as the user query.

Taxonomy:
%s

Return JSON:
{"intent":"general|specific|comparison|troubleshooting","semantic_query":"rich natural-language rewrite","keyword_query":"compact nouns/entities","type_slugs":[],"tags":[],"keywords":[],"attributes":{}}

Rules:
- type_slugs/tags/attribute_keys: only values from taxonomy above; empty if unsure
- keywords: hints for lexical search, not strict filters
- attributes: explicit constraints only (key:value); use only known attribute_keys
- If uncertain, leave lists empty and attributes as {}

Query: %s`, taxonomyBlock, query)

	temp := 0.0
	maxTokens := 500
	messages := []providercore.Message{}
	if strings.TrimSpace(a.cfg.SystemPrompt) != "" {
		messages = append(messages, providercore.Message{Role: "system", Content: a.cfg.SystemPrompt})
	}
	messages = append(messages, providercore.Message{Role: "user", Content: prompt})

	resp, err := a.cfg.LLMProvider.ChatCompletion(ctx, providercore.ChatRequest{
		Model:       a.cfg.QueryAnalyzerModel,
		Temperature: &temp,
		MaxTokens:   &maxTokens,
		Messages:    messages,
		JSONMode:    true,
	})
	if err != nil {
		a.log().Warn("query analyzer failed, using raw query", telemetry.Err(err))
		return &QueryFilters{Intent: "general", SemanticQuery: query, KeywordQuery: query, SearchText: query}, query, query
	}

	respText := jsonutil.ExtractJSON(jsonutil.SanitizeJSON(resp.Content))

	var filters QueryFilters
	if err := json.Unmarshal([]byte(respText), &filters); err != nil {
		a.log().Warn("failed to parse query analyzer response", telemetry.Err(err), telemetry.String("response", respText))
		return &QueryFilters{Intent: "general", SemanticQuery: query, KeywordQuery: query, SearchText: query}, query, query
	}

	if filters.SemanticQuery == "" && filters.SearchText != "" {
		filters.SemanticQuery = filters.SearchText
	}

	semanticText := strings.TrimSpace(filters.SemanticQuery)
	if semanticText == "" {
		semanticText = query
		filters.SemanticQuery = semanticText
	}
	if hasCyrillic(query) && !hasCyrillic(semanticText) {
		semanticText = query
		filters.SemanticQuery = semanticText
	}
	lexicalText := strings.TrimSpace(filters.KeywordQuery)
	if lexicalText == "" {
		lexicalText = semanticText
		filters.KeywordQuery = lexicalText
	}
	filters.SearchText = semanticText

	return &filters, semanticText, lexicalText
}

// DirectAnswerResult is the outcome of the direct-answer gate.
type DirectAnswerResult struct {
	Answer   string
	Hit      bool
	CacheHit ragcache.Tier
}

// TryDirectAnswer asks a cheap LLM whether the query can be answered without
// retrieval. Cached by normalized query + system prompt hash.
func (a *Analyzer) TryDirectAnswer(ctx context.Context, query string) DirectAnswerResult {
	if !shouldRunDirectGate(query, a.cfg.DirectGateMaxWords, a.cfg.DirectGateMaxChars) {
		return DirectAnswerResult{}
	}

	cacheKey := DirectAnswerCacheKey(query, a.cfg.SystemPrompt)
	if a.cfg.DirectAnswerCache != nil {
		if tiered, ok := a.cfg.DirectAnswerCache.(*ragcache.TieredDirectAnswerCache); ok {
			if cached, tier, err := tiered.GetWithTier(ctx, cacheKey); err == nil && tier != ragcache.TierMiss && strings.TrimSpace(cached) != "" {
				return DirectAnswerResult{Answer: cached, Hit: true, CacheHit: tier}
			}
		} else if cached, ok, err := a.cfg.DirectAnswerCache.Get(ctx, cacheKey); err == nil && ok && strings.TrimSpace(cached) != "" {
			return DirectAnswerResult{Answer: cached, Hit: true, CacheHit: ragcache.TierL1}
		}
	}

	prompt := fmt.Sprintf(`Decide whether this user query can be answered directly without any knowledge-base retrieval.

Return ONLY JSON:
{
  "can_answer_immediately": true|false,
  "direct_answer": "string"
}

Rules:
- Set can_answer_immediately=true only for generic small-talk/meta conversational requests that do not require site-specific facts.
- Examples that may be direct: greeting, "who are you", "what can you do", short acknowledgment.
- If any product/service/company-specific details may be needed, set false.
- If can_answer_immediately=false, direct_answer must be empty.
- direct_answer must be in the same language as user query.

User query: %s`, query)

	messages := []providercore.Message{}
	if strings.TrimSpace(a.cfg.SystemPrompt) != "" {
		messages = append(messages, providercore.Message{Role: "system", Content: a.cfg.SystemPrompt})
	}
	messages = append(messages, providercore.Message{Role: "user", Content: prompt})

	temp := 0.0
	maxTokens := 220
	resp, err := a.cfg.LLMProvider.ChatCompletion(ctx, providercore.ChatRequest{
		Model:       a.cfg.QueryAnalyzerModel,
		Temperature: &temp,
		MaxTokens:   &maxTokens,
		Messages:    messages,
		JSONMode:    true,
	})
	if err != nil {
		a.log().Warn("direct-answer checker failed", telemetry.Err(err))
		return DirectAnswerResult{}
	}

	respText := jsonutil.ExtractJSON(jsonutil.SanitizeJSON(resp.Content))
	var parsed struct {
		CanAnswerImmediately bool   `json:"can_answer_immediately"`
		DirectAnswer         string `json:"direct_answer"`
	}
	if err := json.Unmarshal([]byte(respText), &parsed); err != nil {
		a.log().Warn("failed to parse direct-answer checker response", telemetry.Err(err))
		return DirectAnswerResult{}
	}

	answer := strings.TrimSpace(parsed.DirectAnswer)
	if !parsed.CanAnswerImmediately || answer == "" {
		return DirectAnswerResult{}
	}

	if a.cfg.DirectAnswerCache != nil {
		if err := a.cfg.DirectAnswerCache.Set(ctx, cacheKey, answer, a.cfg.DirectCacheTTL); err != nil {
			a.log().Warn("direct-answer cache set failed", telemetry.Err(err))
		}
	}
	return DirectAnswerResult{Answer: answer, Hit: true}
}

func (a *Analyzer) log() telemetry.Logger {
	if a.cfg.Logger == nil {
		return &telemetry.NoOpLogger{}
	}
	return a.cfg.Logger
}

var nonWordRE = regexp.MustCompile(`[^\p{L}\p{N}\s]+`)

// NormalizeDirectQuery is exported because the same normalization is used by
// callers wanting cache-key parity.
func NormalizeDirectQuery(query string) string {
	q := strings.ToLower(strings.TrimSpace(query))
	q = nonWordRE.ReplaceAllString(q, " ")
	q = strings.Join(strings.Fields(q), " ")
	return q
}

// DirectAnswerCacheKey hashes the system prompt so prompt changes invalidate
// the cache automatically.
func DirectAnswerCacheKey(query, systemPrompt string) string {
	normalized := NormalizeDirectQuery(query)
	promptHash := fmt.Sprintf("%x", sha1.Sum([]byte(strings.TrimSpace(systemPrompt))))
	return fmt.Sprintf("v1:%s:%s", promptHash, normalized)
}

func shouldRunDirectGate(query string, maxWords, maxChars int) bool {
	q := NormalizeDirectQuery(query)
	if q == "" {
		return false
	}
	if maxWords <= 0 {
		maxWords = 10
	}
	if maxChars <= 0 {
		maxChars = 120
	}
	if len(q) > maxChars {
		return false
	}
	return len(strings.Fields(q)) <= maxWords
}

func hasCyrillic(s string) bool {
	for _, r := range s {
		if (r >= 'А' && r <= 'я') || r == 'ё' || r == 'Ё' {
			return true
		}
	}
	return false
}
