// Package enricher provides retrieval-time context enrichment: fetching
// full entry text for RAG search results and building context windows.
// This is distinct from ingestion-service/internal/enricher which adds
// semantic metadata (summaries, synthetic questions, tags) to chunks at
// ingest time via LLM.
package enricher

import (
	"context"
	"fmt"
	"strings"

	"github.com/madmike/go-infra/telemetry"
	ragstore "github.com/madmike/go-rag/store"
	"github.com/madmike/go-rag/strategy"
)

// Config holds dependencies for the enricher.
type Config struct {
	Store  ragstore.Store
	Logger telemetry.Logger
}

// Enricher converts raw entry search results into context strings ready for an
// LLM, including parent-hierarchy resolution and structured-attribute formatting.
type Enricher struct {
	cfg Config
}

func New(cfg Config) *Enricher { return &Enricher{cfg: cfg} }

// EnrichToChunks resolves full Entry rows + parents and returns the formatted
// chunks the strategy will return upstream.
func (e *Enricher) EnrichToChunks(ctx context.Context, results []ragstore.SearchResult) ([]strategy.RetrievedChunk, error) {
	if len(results) == 0 {
		return nil, nil
	}

	entryIDs := make([]string, len(results))
	for i, r := range results {
		entryIDs[i] = r.ID
	}

	entries, err := e.cfg.Store.GetEntries(ctx, entryIDs)
	if err != nil {
		e.log().Warn("get entries by ids failed, falling back to summary-only chunks", telemetry.Err(err))
		return chunksFromResults(results), nil
	}

	entryMap := make(map[string]ragstore.Entry, len(entries))
	for _, ent := range entries {
		entryMap[ent.ID] = ent
	}

	parentIDSet := make(map[string]struct{})
	for _, ent := range entries {
		if ent.ParentID != nil && *ent.ParentID != "" {
			parentIDSet[*ent.ParentID] = struct{}{}
		}
	}

	parentMap := make(map[string]ragstore.Entry)
	if len(parentIDSet) > 0 {
		ids := make([]string, 0, len(parentIDSet))
		for id := range parentIDSet {
			ids = append(ids, id)
		}
		parents, err := e.cfg.Store.GetEntries(ctx, ids)
		if err == nil {
			for _, p := range parents {
				parentMap[p.ID] = p
			}
		}
	}

	chunks := make([]strategy.RetrievedChunk, 0, len(results))
	for _, r := range results {
		ent, ok := entryMap[r.ID]
		if !ok {
			chunks = append(chunks, chunkFromResult(r))
			continue
		}
		chunks = append(chunks, formatEntryChunk(ent, parentMap, r.Score))
	}
	return chunks, nil
}

// FormatContext renders the chunks into the same "=== TYPE: NAME ===" sections
// used by the previous enrichment stage, ready to inject into the
// LLM prompt.
func FormatContext(chunks []strategy.RetrievedChunk) string {
	if len(chunks) == 0 {
		return ""
	}
	parts := make([]string, 0, len(chunks))
	for _, c := range chunks {
		var b strings.Builder
		fmt.Fprintf(&b, "=== %s: %s ===\n", strings.ToUpper(c.TypeSlug), c.Name)
		if c.Summary != "" {
			fmt.Fprintf(&b, "Summary: %s\n", c.Summary)
		}
		if c.Content != "" {
			fmt.Fprintf(&b, "Content: %s\n", c.Content)
		}
		if len(c.Attributes) > 0 {
			b.WriteString("Details:\n")
			for k, v := range c.Attributes {
				fmt.Fprintf(&b, "  - %s: %v\n", k, v)
			}
		}
		if len(c.Tags) > 0 {
			fmt.Fprintf(&b, "Categories: %s\n", strings.Join(c.Tags, ", "))
		}
		if len(c.Keywords) > 0 {
			fmt.Fprintf(&b, "Keywords: %s\n", strings.Join(c.Keywords, ", "))
		}
		if c.URL != "" {
			fmt.Fprintf(&b, "More info: %s\n", c.URL)
		}
		parts = append(parts, b.String())
	}
	return strings.Join(parts, "\n")
}

func formatEntryChunk(ent ragstore.Entry, parentMap map[string]ragstore.Entry, score float64) strategy.RetrievedChunk {
	var b strings.Builder
	if ent.ParentID != nil {
		if parent, ok := parentMap[*ent.ParentID]; ok {
			fmt.Fprintf(&b, "Part of: %s (%s)\n", parent.Name, parent.TypeSlug)
		}
	}
	if ent.Content != "" {
		b.WriteString(ent.Content)
	}
	if len(ent.SyntheticQueries) > 0 {
		fmt.Fprintf(&b, "\nCan answer: %s", strings.Join(ent.SyntheticQueries, " | "))
	}
	return strategy.RetrievedChunk{
		ID:         ent.ID,
		Name:       ent.Name,
		TypeSlug:   ent.TypeSlug,
		Summary:    ent.Summary,
		Content:    strings.TrimSpace(b.String()),
		URL:        ent.URL,
		Tags:       ent.Tags,
		Keywords:   ent.Keywords,
		Attributes: ent.Attributes,
		ParentID:   ent.ParentID,
		Score:      score,
	}
}

func chunksFromResults(results []ragstore.SearchResult) []strategy.RetrievedChunk {
	out := make([]strategy.RetrievedChunk, 0, len(results))
	for _, r := range results {
		out = append(out, chunkFromResult(r))
	}
	return out
}

func chunkFromResult(r ragstore.SearchResult) strategy.RetrievedChunk {
	return strategy.RetrievedChunk{
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
		Score:      r.Score,
	}
}

func (e *Enricher) log() telemetry.Logger {
	if e.cfg.Logger == nil {
		return &telemetry.NoOpLogger{}
	}
	return e.cfg.Logger
}
