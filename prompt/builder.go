package prompt

import (
	"bytes"
	"fmt"
	"sort"
	"strings"
	"text/template"

	"github.com/madmike/go-rag/strategy"
)

// Builder defines the dynamic prompt builder for RAG contexts
type Builder struct {
	baseTemplate *template.Template
}

var funcMap = template.FuncMap{
	"formatAttrs": formatAttrs,
	"hasURL": func(url string) bool {
		return url != ""
	},
}

// NewBuilder initializes a Builder with default templates
func NewBuilder() *Builder {
	tmpl := `You have access to the following retrieved knowledge context:
<context>
{{- range $i, $ch := .Chunks }}

[{{ inc $i }}] {{ $ch.Name }}{{ if $ch.URL }} — {{ $ch.URL }}{{ end }}
{{- if $ch.ParentChunk }}
Section: {{ $ch.ParentChunk.Name }}{{ if $ch.ParentChunk.Summary }} — {{ $ch.ParentChunk.Summary }}{{ end }}
{{- end }}
{{ $ch.Content }}
{{- if $ch.Attributes }}
Attributes: {{ formatAttrs $ch.Attributes }}
{{- end }}
{{- end }}
</context>

Use the provided context to answer the user's query. When referencing a specific source, cite it as [N].
If the context does not contain the answer, say so clearly rather than guessing.`

	funcMap["inc"] = func(i int) int { return i + 1 }
	t := template.Must(template.New("rag_context").Funcs(funcMap).Parse(tmpl))
	return &Builder{
		baseTemplate: t,
	}
}

// BuildContext builds the system prompt addition from retrieved chunks
func (b *Builder) BuildContext(result *strategy.Result) (string, error) {
	if result == nil || len(result.Chunks) == 0 {
		return "", nil
	}

	var buf bytes.Buffer
	if err := b.baseTemplate.Execute(&buf, result); err != nil {
		return "", fmt.Errorf("failed to build prompt context: %w", err)
	}

	return strings.TrimSpace(buf.String()), nil
}

// formatAttrs renders a map[string]any as a sorted "key: value" comma list.
func formatAttrs(attrs map[string]any) string {
	if len(attrs) == 0 {
		return ""
	}
	keys := make([]string, 0, len(attrs))
	for k := range attrs {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	var parts []string
	for _, k := range keys {
		parts = append(parts, fmt.Sprintf("%s: %v", k, attrs[k]))
	}
	return strings.Join(parts, ", ")
}
