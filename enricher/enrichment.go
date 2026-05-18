package enricher

import (
	"context"
	"fmt"
	"strings"

	"github.com/madmike/go-infra/telemetry"
	pipelinecore "github.com/madmike/go-pipeline/core"
	ragstore "github.com/madmike/go-rag/store"
)

// EnrichmentStageConfig contains the configuration for the enrichment stage
type EnrichmentStageConfig struct {
	Store  ragstore.Store
	Logger telemetry.Logger
}

// EnrichmentStage enriches RAG results with full entry details and formats context for LLM
type EnrichmentStage struct {
	config EnrichmentStageConfig
}

// NewEnrichmentStage creates a new enrichment stage for entries
func NewEnrichmentStage(config EnrichmentStageConfig) *EnrichmentStage {
	return &EnrichmentStage{
		config: config,
	}
}

// Name returns the stage name
func (s *EnrichmentStage) Name() string {
	return "enrichment"
}

// InputTypes returns the event types this stage accepts
func (s *EnrichmentStage) InputTypes() []pipelinecore.EventType {
	return []pipelinecore.EventType{pipelinecore.EventTypeRAG}
}

// OutputTypes returns the event types this stage produces
func (s *EnrichmentStage) OutputTypes() []pipelinecore.EventType {
	return []pipelinecore.EventType{pipelinecore.EventTypeLLM, pipelinecore.EventTypeError}
}

// Process enriches RAG results with full entry metadata
func (s *EnrichmentStage) Process(ctx context.Context, input <-chan pipelinecore.Event, output chan<- pipelinecore.Event) error {
	logger := s.config.Logger.WithModule(s.Name())
	logger.Info("EnrichmentStage started processing")

	// Process all RAG events
	for event := range input {
		ragEvent, ok := event.(pipelinecore.RAGEvent)
		if !ok {
			// Pass through non-RAG events
			output <- event
			continue
		}

		// Handle fallback case
		if len(ragEvent.Results) == 0 {
			if msg, ok := ragEvent.Metadata["message"].(string); ok {
				output <- pipelinecore.LLMEvent{
					Content: ragEvent.Query + "\n\nContext: " + msg,
				}
			} else {
				output <- pipelinecore.LLMEvent{
					Content: ragEvent.Query,
				}
			}
			continue
		}

		logger.Info("Enriching RAG results", telemetry.Int("count", len(ragEvent.Results)))

		// Extract entry IDs from RAG results
		entryIDs := make([]string, len(ragEvent.Results))
		for i, result := range ragEvent.Results {
			entryIDs[i] = result.DocumentID
		}

		// Fetch full entry details
		entries, err := s.config.Store.GetEntries(ctx, entryIDs)
		if err != nil {
			logger.Error("Failed to fetch entry details", telemetry.Err(err))
			// Fallback: use summaries from RAG results
			s.enrichFromRAGResults(ragEvent, output)
			continue
		}

		// Build entry map for quick lookup
		entryMap := make(map[string]ragstore.Entry)
		for _, entry := range entries {
			entryMap[entry.ID] = entry
		}

		// Get parent entries for hierarchy context
		parentIDs := make(map[string]bool)
		for _, entry := range entries {
			if entry.ParentID != nil && *entry.ParentID != "" {
				parentIDs[*entry.ParentID] = true
			}
		}

		var parentEntries []ragstore.Entry
		if len(parentIDs) > 0 {
			parentIDList := make([]string, 0, len(parentIDs))
			for id := range parentIDs {
				parentIDList = append(parentIDList, id)
			}
			parentEntries, _ = s.config.Store.GetEntries(ctx, parentIDList)
		}

		parentMap := make(map[string]ragstore.Entry)
		for _, parent := range parentEntries {
			parentMap[parent.ID] = parent
		}

		// Format enriched context
		var contextParts []string

		// Add each entry with rich metadata
		for _, ragResult := range ragEvent.Results {
			entry, ok := entryMap[ragResult.DocumentID]
			if !ok {
				continue
			}

			var entryContext strings.Builder
			entryContext.WriteString(fmt.Sprintf("=== %s: %s ===\n", strings.ToUpper(entry.TypeSlug), entry.Name))

			// Add parent hierarchy if exists
			if entry.ParentID != nil {
				if parent, ok := parentMap[*entry.ParentID]; ok {
					entryContext.WriteString(fmt.Sprintf("Part of: %s (%s)\n", parent.Name, parent.TypeSlug))
				}
			}

			// Add summary
			if entry.Summary != "" {
				entryContext.WriteString(fmt.Sprintf("Summary: %s\n", entry.Summary))
			}

			// Add main content
			if entry.Content != "" {
				entryContext.WriteString(fmt.Sprintf("Content: %s\n", entry.Content))
			}

			// Add structured attributes
			if len(entry.Attributes) > 0 {
				entryContext.WriteString("Details:\n")
				for key, value := range entry.Attributes {
					entryContext.WriteString(fmt.Sprintf("  - %s: %v\n", key, value))
				}
			}

			// Add tags for categorization
			if len(entry.Tags) > 0 {
				entryContext.WriteString(fmt.Sprintf("Categories: %s\n", strings.Join(entry.Tags, ", ")))
			}

			// Add retrieval keywords to improve downstream grounding.
			if len(entry.Keywords) > 0 {
				entryContext.WriteString(fmt.Sprintf("Keywords: %s\n", strings.Join(entry.Keywords, ", ")))
			}

			// Add synthetic questions this entry can answer.
			if len(entry.SyntheticQueries) > 0 {
				entryContext.WriteString(fmt.Sprintf("Can answer: %s\n", strings.Join(entry.SyntheticQueries, " | ")))
			}

			// Add URL for reference
			if entry.URL != "" {
				entryContext.WriteString(fmt.Sprintf("More info: %s\n", entry.URL))
			}

			contextParts = append(contextParts, entryContext.String())
		}

		enrichedContext := strings.Join(contextParts, "\n")

		logger.Debug("Enrichment complete", telemetry.Int("context_length", len(enrichedContext)))

		// Emit LLM event with enriched context
		output <- pipelinecore.LLMEvent{
			Content: fmt.Sprintf("%s\n\n=== User Question ===\n%s", enrichedContext, ragEvent.Query),
		}
	}

	return nil
}

// enrichFromRAGResults falls back to using summaries from RAG results when full fetch fails
func (s *EnrichmentStage) enrichFromRAGResults(ragEvent pipelinecore.RAGEvent, output chan<- pipelinecore.Event) {
	var contextParts []string

	for _, result := range ragEvent.Results {
		var entryContext strings.Builder

		if name, ok := result.Metadata["name"].(string); ok {
			typeSlug, _ := result.Metadata["type_slug"].(string)
			entryContext.WriteString(fmt.Sprintf("=== %s: %s ===\n", strings.ToUpper(typeSlug), name))
		}

		entryContext.WriteString(result.Content)
		contextParts = append(contextParts, entryContext.String())
	}

	enrichedContext := strings.Join(contextParts, "\n\n")

	output <- pipelinecore.LLMEvent{
		Content: fmt.Sprintf("%s\n\n=== User Question ===\n%s", enrichedContext, ragEvent.Query),
	}
}
