package strategy

import (
	"context"

	ragstore "github.com/madmike/go-rag/store"
)

// isStructuredIntent returns true for intents that require KB lookup and must
// not be short-circuited by the direct-answer gate (to prevent hallucination
// on product/service/booking queries).
func isStructuredIntent(intent string) bool {
	switch intent {
	case "product_search", "service_lookup", "booking":
		return true
	}
	return false
}

// resolveParents bulk-fetches the parent section entries for any chunks that
// have a ParentID, then attaches them as ParentChunk for Small-to-Big retrieval.
// Chunks that are already section nodes (no ParentID) are left unchanged.
func resolveParents(ctx context.Context, store ragstore.Store, chunks []RetrievedChunk) []RetrievedChunk {
	parentIDs := collectParentIDs(chunks)
	if len(parentIDs) == 0 {
		return chunks
	}

	entries, err := store.GetEntries(ctx, parentIDs)
	if err != nil || len(entries) == 0 {
		return chunks
	}

	byID := make(map[string]ragstore.Entry, len(entries))
	for _, e := range entries {
		byID[e.ID] = e
	}

	out := make([]RetrievedChunk, len(chunks))
	for i, ch := range chunks {
		if ch.ParentID == nil {
			out[i] = ch
			continue
		}
		if parent, ok := byID[*ch.ParentID]; ok {
			parent := entryToChunk(parent)
			ch.ParentChunk = &parent
		}
		out[i] = ch
	}
	return out
}

func collectParentIDs(chunks []RetrievedChunk) []string {
	seen := make(map[string]bool)
	var ids []string
	for _, ch := range chunks {
		if ch.ParentID != nil && !seen[*ch.ParentID] {
			seen[*ch.ParentID] = true
			ids = append(ids, *ch.ParentID)
		}
	}
	return ids
}

func entryToChunk(e ragstore.Entry) RetrievedChunk {
	return RetrievedChunk{
		ID:         e.ID,
		Name:       e.Name,
		TypeSlug:   e.TypeSlug,
		Summary:    e.Summary,
		Content:    e.Content,
		URL:        e.URL,
		Tags:       e.Tags,
		Keywords:   e.Keywords,
		Attributes: e.Attributes,
		ParentID:   e.ParentID,
	}
}
