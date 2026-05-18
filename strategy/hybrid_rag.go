package strategy

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
	ragstore "github.com/madmike/go-rag/store"
)

// ragStepTimer is a helper that records step latency as a trace note.
// It also emits a Prometheus observation when the agent-runtime telemetry
// package is not directly importable from this library.
// Each step label must match the agent_rag_step_latency_ms metric labels.
func ragStepNote(trace *RetrievalTrace, step string, ms int64) {
	trace.Notes = append(trace.Notes, fmt.Sprintf("%s:%dms", step, ms))
}

// DirectAnswerCache interface
type DirectAnswerCache interface {
	Get(ctx context.Context, key string) (string, bool, error)
	Set(ctx context.Context, key string, val string, ttl time.Duration) error
}

// HybridStrategyConfig contains the configuration for the hybrid RAG strategy
type HybridStrategyConfig struct {
	Store              ragstore.Store
	LLMProvider        providercore.LLMProvider
	DirectAnswerCache  DirectAnswerCache
	DirectCacheTTL     time.Duration
	DirectGateMaxWords int
	DirectGateMaxChars int
	EmbeddingProvider  providercore.EmbeddingProvider
	EmbeddingModel     string
	QueryAnalyzerModel string
	MaxChunks          int
	VectorWeight       float64
	KeywordWeight      float64
	Threshold          float64
	FallbackContent    string
	Logger             telemetry.Logger
}

// HybridStrategy performs hybrid vector + tsvector + SQL-filtered search on entries
type HybridStrategy struct {
	config HybridStrategyConfig
}

// NewHybridStrategy creates a new hybrid RAG strategy
func NewHybridStrategy(config HybridStrategyConfig) *HybridStrategy {
	return &HybridStrategy{
		config: config,
	}
}

// Name returns the strategy name
func (s *HybridStrategy) Name() string {
	return NameHybrid
}

// Retrieve executes the hybrid retrieval
func (s *HybridStrategy) Retrieve(ctx context.Context, q Query) (*Result, error) {
	logger := s.config.Logger.WithModule(s.Name())
	logger.Info("HybridStrategy started retrieving", telemetry.String("query", q.UserText))
	start := time.Now()

	trace := RetrievalTrace{
		Strategy:  s.Name(),
		StartedAt: start,
	}

	// 1. Direct-answer short-circuit.
	// Skipped when: structured-data intents (products/services/bookings require KB facts),
	// OR when the classifier has already decided needs_rag=true (PreAnalyzedFilters set).
	if !isStructuredIntent(q.RAGIntent) && q.PreAnalyzedFilters == nil {
		directGateStart := time.Now()
		if directAnswer, ok := s.tryDirectAnswer(ctx, q.UserText, q.SystemPrompt, &trace); ok {
			ragStepNote(&trace, "direct_gate", time.Since(directGateStart).Milliseconds())
			trace.LatencyMs = time.Since(start).Milliseconds()
			return &Result{
				Mode:         ModeDirectAnswer,
				DirectAnswer: directAnswer,
				Trace:        trace,
			}, nil
		}
		ragStepNote(&trace, "direct_gate", time.Since(directGateStart).Milliseconds())
	}

	// 2. Get taxonomy
	taxonomyStart := time.Now()
	taxonomy, err := s.config.Store.GetTaxonomy(ctx, q.SourceIDs)
	if err != nil {
		logger.Warn("Failed to get source taxonomy", telemetry.Err(err))
		taxonomy = &ragstore.Taxonomy{}
	}
	ragStepNote(&trace, "taxonomy", time.Since(taxonomyStart).Milliseconds())

	// 3. Analyze query — skipped when the classifier already produced filters.
	var filters *QueryFilters
	var semanticSearchText, lexicalSearchText string

	if q.PreAnalyzedFilters != nil {
		// Use pre-analyzed filters from the merged classifier call.
		// Still validate against taxonomy to drop any hallucinated values.
		f := *q.PreAnalyzedFilters
		f.TypeSlugs = intersect(f.TypeSlugs, taxonomy.TypeSlugs)
		if len(f.TypeSlugs) == 0 {
			f.TypeSlugs = nil
		}
		f.Tags = intersect(f.Tags, taxonomy.Tags)
		if len(f.Tags) == 0 {
			f.Tags = nil
		}
		if len(f.Attributes) > 0 && len(taxonomy.AttributeKeys) > 0 {
			allowed := toSet(taxonomy.AttributeKeys)
			for k := range f.Attributes {
				if !allowed[k] {
					delete(f.Attributes, k)
				}
			}
		}
		semanticSearchText = f.SemanticQuery
		if semanticSearchText == "" {
			semanticSearchText = q.UserText
		}
		lexicalSearchText = f.KeywordQuery
		if lexicalSearchText == "" {
			lexicalSearchText = semanticSearchText
		}
		// For procedural/faq/troubleshooting intents, scrub catalog-only type_slugs so document chunks compete.
		if isDocumentFocusedIntent(f.Intent) {
			f.TypeSlugs = scrubCatalogSlugs(f.TypeSlugs)
		}
		filters = &f
		ragStepNote(&trace, "analyze", 0) // 0ms — skipped
		logger.Info("rag_analyze_skipped", telemetry.String("reason", "pre_analyzed"))
	} else {
		analyzeStart := time.Now()
		filters, semanticSearchText, lexicalSearchText = s.analyzeQuery(ctx, q.UserText, taxonomy, q.SystemPrompt, &trace)
		ragStepNote(&trace, "analyze", time.Since(analyzeStart).Milliseconds())
	}

	trace.Intent = filters.Intent
	trace.SemanticQuery = semanticSearchText
	trace.LexicalQuery = lexicalSearchText

	// Merge classifier hint filters when the full analyze ran (not pre-analyzed path).
	if q.PreAnalyzedFilters == nil {
		if len(q.Hints.TypeSlugs) > 0 && len(filters.TypeSlugs) == 0 {
			filters.TypeSlugs = intersect(q.Hints.TypeSlugs, taxonomy.TypeSlugs)
		}
		if len(q.Hints.Tags) > 0 && len(filters.Tags) == 0 {
			filters.Tags = intersect(q.Hints.Tags, taxonomy.Tags)
		}
	}

	filterMap := make(map[string]any)
	if len(filters.TypeSlugs) > 0 {
		filterMap["type_slugs"] = filters.TypeSlugs
	}
	if len(filters.Tags) > 0 {
		filterMap["tags"] = filters.Tags
	}
	if len(filters.Keywords) > 0 {
		filterMap["keywords"] = filters.Keywords
	}
	if len(filters.Attributes) > 0 {
		filterMap["attributes"] = filters.Attributes
	}
	trace.Filters = filterMap

	// 4. Generate Embedding
	embedStart := time.Now()
	embResp, err := s.config.EmbeddingProvider.GenerateEmbedding(ctx, providercore.EmbeddingRequest{
		Text:  semanticSearchText,
		Model: s.config.EmbeddingModel,
	})
	var embedding []float32
	if err == nil && embResp != nil {
		embedding = embResp.Vector
		trace.EmbeddingModel = embResp.Model
		if embResp.Usage != nil {
			trace.InputTokens += embResp.Usage.InputTokens
		} else {
			trace.InputTokens += len(semanticSearchText) / 4
		}
	}
	ragStepNote(&trace, "embed", time.Since(embedStart).Milliseconds())
	logger.Info("rag_embed_done", telemetry.Int64("embed_ms", time.Since(embedStart).Milliseconds()))

	// 5. Execute Hybrid Search
	topK := s.config.MaxChunks
	if q.Hints.TopK > 0 {
		topK = q.Hints.TopK
	}

	threshold := s.config.Threshold
	if q.Hints.Threshold > 0 {
		threshold = q.Hints.Threshold
	}

	baseParams := ragstore.HybridSearchParams{
		TenantID:         q.TenantID,
		QueryEmbedding:   embedding,
		QueryText:        lexicalSearchText,
		SourceIDs:        q.SourceIDs,
		FilterTypeSlugs:  filters.TypeSlugs,
		FilterTags:       filters.Tags,
		FilterAttributes: filters.Attributes,
		MatchCount:       topK,
		VectorWeight:     s.config.VectorWeight,
		KeywordWeight:    s.config.KeywordWeight,
		MatchThreshold:   threshold,
	}

	searchStart := time.Now()
	results, err := s.config.Store.HybridSearch(ctx, baseParams)
	if err != nil {
		logger.Error("Hybrid search failed", telemetry.Err(err))
		trace.LatencyMs = time.Since(start).Milliseconds()
		return nil, fmt.Errorf("hybrid search failed: %w", err)
	}

	// Relax filters if empty or scores are too low
	minScoreThreshold := 0.02
	shouldRelax := len(results) == 0 || (len(results) > 0 && results[0].Score < minScoreThreshold)

	if shouldRelax {
		if len(results) > 0 {
			logger.Info("rag_relax_triggered",
				telemetry.String("reason", "low_score"),
				telemetry.Float64("top_score", results[0].Score),
				telemetry.Float64("threshold", minScoreThreshold))
		}
		relaxed := baseParams
		relaxed.FilterAttributes = nil
		results, _ = s.config.Store.HybridSearch(ctx, relaxed)

		if len(results) == 0 || (len(results) > 0 && results[0].Score < minScoreThreshold) {
			relaxed.FilterTypeSlugs = nil
			relaxed.FilterTags = nil
			results, _ = s.config.Store.HybridSearch(ctx, relaxed)
		}
	}
	ragStepNote(&trace, "search", time.Since(searchStart).Milliseconds())
	logger.Info("rag_search_done",
		telemetry.Int("results", len(results)),
		telemetry.Int64("search_ms", time.Since(searchStart).Milliseconds()),
	)

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

	parentsStart := time.Now()
	chunks = resolveParents(ctx, s.config.Store, chunks)
	ragStepNote(&trace, "parents", time.Since(parentsStart).Milliseconds())

	trace.ChunksCount = len(chunks)
	trace.LatencyMs = time.Since(start).Milliseconds()
	logger.Info("rag_retrieve_done",
		telemetry.Int("chunks", len(chunks)),
		telemetry.Int64("total_ms", trace.LatencyMs),
		telemetry.String("notes", strings.Join(trace.Notes, " ")),
	)

	return &Result{
		Mode:   ModeHybrid,
		Chunks: chunks,
		Trace:  trace,
	}, nil
}

func (s *HybridStrategy) tryDirectAnswer(ctx context.Context, query, systemPrompt string, trace *RetrievalTrace) (string, bool) {
	if !shouldRunDirectGate(query, s.config.DirectGateMaxWords, s.config.DirectGateMaxChars) {
		return "", false
	}

	cacheKey := directAnswerCacheKey(query, systemPrompt)
	if s.config.DirectAnswerCache != nil {
		if cached, ok, _ := s.config.DirectAnswerCache.Get(ctx, cacheKey); ok && strings.TrimSpace(cached) != "" {
			trace.CacheHit = "l1" // simplistic designation
			return cached, true
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
	if strings.TrimSpace(systemPrompt) != "" {
		messages = append(messages, providercore.Message{
			Role:    "system",
			Content: systemPrompt,
		})
	}
	messages = append(messages, providercore.Message{Role: "user", Content: prompt})

	temp := 0.0
	maxTokens := 220
	resp, err := s.config.LLMProvider.ChatCompletion(ctx, providercore.ChatRequest{
		Model:       s.config.QueryAnalyzerModel,
		Temperature: &temp,
		MaxTokens:   &maxTokens,
		Messages:    messages,
		JSONMode:    true,
	})
	if err != nil {
		return "", false
	}

	if resp.Usage != nil {
		trace.LLMInputTokens += resp.Usage.InputTokens
		trace.LLMOutputTokens += resp.Usage.OutputTokens
	}

	respText := jsonutil.ExtractJSON(jsonutil.SanitizeJSON(resp.Content))
	var parsed struct {
		CanAnswerImmediately bool   `json:"can_answer_immediately"`
		DirectAnswer         string `json:"direct_answer"`
	}
	if err := json.Unmarshal([]byte(respText), &parsed); err != nil {
		return "", false
	}

	answer := strings.TrimSpace(parsed.DirectAnswer)
	if parsed.CanAnswerImmediately && answer != "" {
		if s.config.DirectAnswerCache != nil {
			_ = s.config.DirectAnswerCache.Set(ctx, cacheKey, answer, s.config.DirectCacheTTL)
		}
		return answer, true
	}
	return "", false
}

var nonWordRE = regexp.MustCompile(`[^\p{L}\p{N}\s]+`)

func normalizeDirectQuery(query string) string {
	q := strings.ToLower(strings.TrimSpace(query))
	q = nonWordRE.ReplaceAllString(q, " ")
	q = strings.Join(strings.Fields(q), " ")
	return q
}

func directAnswerCacheKey(query, systemPrompt string) string {
	normalized := normalizeDirectQuery(query)
	promptHash := fmt.Sprintf("%x", sha1.Sum([]byte(strings.TrimSpace(systemPrompt))))
	return fmt.Sprintf("v1:%s:%s", promptHash, normalized)
}

func shouldRunDirectGate(query string, maxWords, maxChars int) bool {
	q := normalizeDirectQuery(query)
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

// QueryFilters holds extracted filters from query analysis
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

func (s *HybridStrategy) analyzeQuery(ctx context.Context, query string, taxonomy *ragstore.Taxonomy, systemPrompt string, trace *RetrievalTrace) (*QueryFilters, string, string) {
	attrSchemaBytes, _ := json.Marshal(taxonomy.AttributeSchema)
	attrSchemaJSON := string(attrSchemaBytes)
	if attrSchemaJSON == "" || attrSchemaJSON == "null" {
		attrSchemaJSON = "{}"
	}

	prompt := fmt.Sprintf(`Plan a hybrid RAG query. Write semantic_query and keyword_query in the same language as the user query.

Taxonomy:
  types: %s
  tags: %s
  attributes: %s

Return JSON:
{"intent":"general|specific|comparison|troubleshooting","semantic_query":"rich natural-language rewrite","keyword_query":"compact nouns/entities","type_slugs":[],"tags":[],"keywords":[],"attributes":{}}

Attribute operators (use only keys from the schema above):
  numeric:     {"k":{"$gte":N}} ≥N | {"k":{"$lte":N}} ≤N | {"k":{"$gt":N}} >N | {"k":{"$lt":N}} <N | {"k":N} exact
  categorical: {"k":"ExactValue"} — copy value exactly from schema
  text:        {"k":{"$contains":"fragment"}} case-insensitive partial match
  set:         {"k":{"$in":["v1","v2"]}}

Price/duration: when schema has min/max price keys, apply floor/ceiling based on user intent ("more than X"→$gte on min key, "less than X"→$lte on max key, range→both). Apply similarly for duration keys.

If unsure, leave lists empty and attributes {}.

Query: %s`,
		strings.Join(taxonomy.TypeSlugs, ", "),
		strings.Join(taxonomy.Tags, ", "),
		attrSchemaJSON,
		query,
	)

	temp := 0.0
	maxTokens := 500
	messages := []providercore.Message{}
	if strings.TrimSpace(systemPrompt) != "" {
		messages = append(messages, providercore.Message{
			Role:    "system",
			Content: systemPrompt,
		})
	}
	messages = append(messages, providercore.Message{
		Role:    "user",
		Content: prompt,
	})
	resp, err := s.config.LLMProvider.ChatCompletion(ctx, providercore.ChatRequest{
		Model:       s.config.QueryAnalyzerModel,
		Temperature: &temp,
		MaxTokens:   &maxTokens,
		Messages:    messages,
		JSONMode:    true,
	})

	if err != nil {
		return &QueryFilters{
			Intent:        "general",
			SemanticQuery: query,
			KeywordQuery:  query,
			SearchText:    query,
		}, query, query
	}

	if resp.Usage != nil {
		trace.LLMInputTokens += resp.Usage.InputTokens
		trace.LLMOutputTokens += resp.Usage.OutputTokens
	}

	respText := jsonutil.ExtractJSON(jsonutil.SanitizeJSON(resp.Content))
	var filters QueryFilters
	if err := json.Unmarshal([]byte(respText), &filters); err != nil {
		return &QueryFilters{
			Intent:        "general",
			SemanticQuery: query,
			KeywordQuery:  query,
			SearchText:    query,
		}, query, query
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

	// Validate extracted filters against taxonomy — drop unknown values to
	// avoid spurious SQL constraints that return zero results.
	filters.TypeSlugs = intersect(filters.TypeSlugs, taxonomy.TypeSlugs)
	if len(filters.TypeSlugs) == 0 {
		filters.TypeSlugs = nil
	}
	filters.Tags = intersect(filters.Tags, taxonomy.Tags)
	if len(filters.Tags) == 0 {
		filters.Tags = nil
	}
	if len(filters.Attributes) > 0 && len(taxonomy.AttributeKeys) > 0 {
		allowed := toSet(taxonomy.AttributeKeys)
		for k := range filters.Attributes {
			if !allowed[k] {
				delete(filters.Attributes, k)
			}
		}
	}

	return &filters, semanticText, lexicalText
}

// intersect returns elements of a that are also in b.
// Returns nil (not empty slice) when a is empty, preserving "no filter" semantics.
func intersect(a, b []string) []string {
	if len(a) == 0 {
		return nil
	}
	set := toSet(b)
	var out []string
	for _, v := range a {
		if set[v] {
			out = append(out, v)
		}
	}
	return out
}

func toSet(s []string) map[string]bool {
	m := make(map[string]bool, len(s))
	for _, v := range s {
		m[v] = true
	}
	return m
}

func isDocumentFocusedIntent(intent string) bool {
	return intent == "faq" || intent == "procedure" || intent == "troubleshooting"
}

func scrubCatalogSlugs(slugs []string) []string {
	if len(slugs) == 0 {
		return nil
	}
	catalogSlugs := toSet([]string{"service", "product"})
	var out []string
	for _, slug := range slugs {
		if !catalogSlugs[slug] {
			out = append(out, slug)
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func hasCyrillic(s string) bool {
	for _, r := range s {
		if (r >= 'А' && r <= 'я') || r == 'ё' || r == 'Ё' {
			return true
		}
	}
	return false
}
