package prompt

import (
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/madmike/go-rag/strategy"
)

// TestBuilderCreation initializes builder with default template.
func TestBuilderCreation(t *testing.T) {
	builder := NewBuilder()
	require.NotNil(t, builder)
	require.NotNil(t, builder.baseTemplate)
}

// TestBuildContextEmpty handles nil result.
func TestBuildContextEmpty(t *testing.T) {
	builder := NewBuilder()

	context, err := builder.BuildContext(nil)
	require.NoError(t, err)
	require.Equal(t, "", context)
}

// TestBuildContextNoChunks handles empty chunks.
func TestBuildContextNoChunks(t *testing.T) {
	builder := NewBuilder()
	result := &strategy.Result{Chunks: []strategy.RetrievedChunk{}}

	context, err := builder.BuildContext(result)
	require.NoError(t, err)
	require.Equal(t, "", context)
}

// TestBuildContextSingleChunk renders one chunk.
func TestBuildContextSingleChunk(t *testing.T) {
	builder := NewBuilder()
	result := &strategy.Result{
		Chunks: []strategy.RetrievedChunk{
			{
				Name:    "Article Title",
				Content: "Article body content here",
				URL:     "https://example.com/article",
			},
		},
	}

	context, err := builder.BuildContext(result)
	require.NoError(t, err)
	require.Contains(t, context, "Article Title")
	require.Contains(t, context, "Article body content here")
	require.Contains(t, context, "https://example.com/article")
	require.Contains(t, context, "[1]")
}

// TestBuildContextMultipleChunks indexes and separates chunks.
func TestBuildContextMultipleChunks(t *testing.T) {
	builder := NewBuilder()
	result := &strategy.Result{
		Chunks: []strategy.RetrievedChunk{
			{Name: "First", Content: "First content"},
			{Name: "Second", Content: "Second content"},
			{Name: "Third", Content: "Third content"},
		},
	}

	context, err := builder.BuildContext(result)
	require.NoError(t, err)
	require.Contains(t, context, "[1] First")
	require.Contains(t, context, "[2] Second")
	require.Contains(t, context, "[3] Third")
}

// TestBuildContextWithAttributes includes formatted attributes.
func TestBuildContextWithAttributes(t *testing.T) {
	builder := NewBuilder()
	result := &strategy.Result{
		Chunks: []strategy.RetrievedChunk{
			{
				Name:       "Product",
				Content:    "Widget details",
				Attributes: map[string]any{"price": 29.99, "color": "blue"},
			},
		},
	}

	context, err := builder.BuildContext(result)
	require.NoError(t, err)
	require.Contains(t, context, "Attributes:")
	require.Contains(t, context, "price:")
	require.Contains(t, context, "color:")
}

// TestBuildContextWithParentChunk includes section info.
func TestBuildContextWithParentChunk(t *testing.T) {
	builder := NewBuilder()
	parentChunk := strategy.RetrievedChunk{
		Name:    "Main Topic",
		Summary: "Overview",
	}
	result := &strategy.Result{
		Chunks: []strategy.RetrievedChunk{
			{
				Name:        "Subsection",
				Content:     "Details here",
				ParentChunk: &parentChunk,
			},
		},
	}

	context, err := builder.BuildContext(result)
	require.NoError(t, err)
	require.Contains(t, context, "Section:")
	require.Contains(t, context, "Main Topic")
}

// TestBuildContextURL omits missing URLs.
func TestBuildContextURLHandling(t *testing.T) {
	builder := NewBuilder()
	result := &strategy.Result{
		Chunks: []strategy.RetrievedChunk{
			{
				Name:    "No URL",
				Content: "Content",
			},
			{
				Name:    "With URL",
				Content: "Content",
				URL:     "https://example.com",
			},
		},
	}

	context, err := builder.BuildContext(result)
	require.NoError(t, err)
	require.Contains(t, context, "[2] With URL")
	require.Contains(t, context, "https://example.com")
	require.NotContains(t, context, "— —")
}

// TestFormatAttrsEmpty handles nil or empty map.
func TestFormatAttrsEmpty(t *testing.T) {
	result := formatAttrs(nil)
	require.Equal(t, "", result)

	result = formatAttrs(map[string]any{})
	require.Equal(t, "", result)
}

// TestFormatAttrsSort sorts keys alphabetically.
func TestFormatAttrsSort(t *testing.T) {
	attrs := map[string]any{
		"zebra": "z",
		"apple": "a",
		"melon": "m",
	}

	result := formatAttrs(attrs)
	appleIdx := indexOfSubstring(result, "apple:")
	melonIdx := indexOfSubstring(result, "melon:")
	zebraIdx := indexOfSubstring(result, "zebra:")

	require.True(t, appleIdx < melonIdx)
	require.True(t, melonIdx < zebraIdx)
}

// TestFormatAttrsTypes handles various value types.
func TestFormatAttrsTypes(t *testing.T) {
	attrs := map[string]any{
		"string":  "value",
		"number":  42,
		"float":   3.14,
		"boolean": true,
	}

	result := formatAttrs(attrs)
	require.Contains(t, result, "string: value")
	require.Contains(t, result, "number: 42")
	require.Contains(t, result, "float: 3.14")
	require.Contains(t, result, "boolean: true")
}

// TestBuildContextCommonFields checks standard context structure.
func TestBuildContextCommonFields(t *testing.T) {
	builder := NewBuilder()
	result := &strategy.Result{
		Chunks: []strategy.RetrievedChunk{
			{Name: "Test", Content: "Body"},
		},
	}

	context, err := builder.BuildContext(result)
	require.NoError(t, err)
	require.Contains(t, context, "retrieved knowledge context")
	require.Contains(t, context, "cite it as [N]")
	require.Contains(t, context, "<context>")
}

// TestBuildContextCitationInstruction ensures answer guidance is present.
func TestBuildContextCitationInstruction(t *testing.T) {
	builder := NewBuilder()
	result := &strategy.Result{
		Chunks: []strategy.RetrievedChunk{
			{Name: "Source", Content: "Answer"},
		},
	}

	context, err := builder.BuildContext(result)
	require.NoError(t, err)
	require.Contains(t, context, "[1]")
	require.Contains(t, context, "context does not contain the answer")
}

// TestBuildContextWhitespace trims results.
func TestBuildContextWhitespace(t *testing.T) {
	builder := NewBuilder()
	result := &strategy.Result{
		Chunks: []strategy.RetrievedChunk{
			{Name: "Test", Content: "Body"},
		},
	}

	context, err := builder.BuildContext(result)
	require.NoError(t, err)
	require.Equal(t, context, string(context[:]))
	require.False(t, len(context) > 0 && context[0] == ' ')
}

// TestFormatAttrsMultiple renders multiple attributes.
func TestFormatAttrsMultiple(t *testing.T) {
	attrs := map[string]any{
		"category":     "electronics",
		"manufacturer": "ACME",
	}

	result := formatAttrs(attrs)
	require.Contains(t, result, "category: electronics")
	require.Contains(t, result, "manufacturer: ACME")
	require.Contains(t, result, ",")
}

// TestBuildContextInternals checks template execution.
func TestBuildContextInternals(t *testing.T) {
	builder := NewBuilder()
	result := &strategy.Result{
		Chunks: []strategy.RetrievedChunk{
			{
				Name:       "Product",
				Content:    "Details",
				Attributes: map[string]any{"in_stock": true},
				URL:        "https://shop.local/item",
			},
		},
	}

	context, err := builder.BuildContext(result)
	require.NoError(t, err)
	require.NotEmpty(t, context)
	require.Greater(t, len(context), 50)
}

func indexOfSubstring(s, substr string) int {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return i
		}
	}
	return -1
}
