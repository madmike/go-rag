package strategy

import (
	"context"
	"time"
)

// None is a no-op strategy used for small-talk and tool-driven workflows that
// do not need knowledge-base retrieval.
type None struct{}

func NewNone() *None { return &None{} }

func (n *None) Name() string { return NameNone }

func (n *None) Retrieve(ctx context.Context, q Query) (*Result, error) {
	start := time.Now()
	return &Result{
		Mode: ModeSkipped,
		Trace: RetrievalTrace{
			Strategy:  NameNone,
			StartedAt: start,
			LatencyMs: time.Since(start).Milliseconds(),
		},
	}, nil
}
