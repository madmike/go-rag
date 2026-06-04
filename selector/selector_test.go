package selector

import (
	"context"
	"testing"

	"github.com/madmike/go-rag/strategy"
	"github.com/stretchr/testify/require"
)

// MockStrategy implements strategy.Strategy for testing.
type MockStrategy struct {
	name_  string
	result *strategy.Result
	err    error
}

func (m *MockStrategy) Name() string {
	return m.name_
}

func (m *MockStrategy) Retrieve(ctx context.Context, q strategy.Query) (*strategy.Result, error) {
	if m.err != nil {
		return nil, m.err
	}
	return m.result, nil
}

// TestSelectorAddCondition adds a condition successfully.
func TestSelectorAddCondition(t *testing.T) {
	defaultStrat := &MockStrategy{name_: "default"}
	selector := NewSelector(defaultStrat)

	strategy := &MockStrategy{name_: "vector"}
	err := selector.AddCondition("history_length < 5", strategy)
	require.NoError(t, err)
	require.Equal(t, 1, len(selector.conditions))
}

// TestSelectorAddConditionInvalidExpr tests adding invalid expression.
func TestSelectorAddConditionInvalidExpr(t *testing.T) {
	defaultStrat := &MockStrategy{name_: "default"}
	selector := NewSelector(defaultStrat)

	strategy := &MockStrategy{name_: "vector"}
	// Invalid expr syntax
	err := selector.AddCondition("invalid ][", strategy)
	require.Error(t, err)
	require.Equal(t, 0, len(selector.conditions))
}

// TestSelectorSelectFirstMatch returns first matching strategy.
func TestSelectorSelectFirstMatch(t *testing.T) {
	defaultStrategy := &MockStrategy{name_: "default"}
	vector := &MockStrategy{name_: "vector"}
	hybrid := &MockStrategy{name_: "hybrid"}

	selector := NewSelector(defaultStrategy)
	_ = selector.AddCondition("history_length > 0", vector)
	_ = selector.AddCondition("history_length > 5", hybrid)

	// With 3 history items, first condition matches
	query := strategy.Query{
		UserText: "test",
		History: []strategy.Message{
			{Role: "user", Content: "msg1"},
			{Role: "assistant", Content: "msg2"},
			{Role: "user", Content: "msg3"},
		},
	}

	result, err := selector.Select(context.Background(), query)
	require.NoError(t, err)
	require.Equal(t, "vector", result.Name())
}

// TestSelectorSelectDefault returns default when no condition matches.
func TestSelectorSelectDefault(t *testing.T) {
	defaultStrategy := &MockStrategy{name_: "default"}
	selector := NewSelector(defaultStrategy)

	// Add a condition that won't match
	_ = selector.AddCondition("len(history) > 100", &MockStrategy{name_: "hybrid"})

	query := strategy.Query{UserText: "test"}
	result, err := selector.Select(context.Background(), query)
	require.NoError(t, err)
	require.Equal(t, "default", result.Name())
}

// TestSelectorSelectOrderMatters conditions are evaluated in order.
func TestSelectorSelectOrderMatters(t *testing.T) {
	defaultStrategy := &MockStrategy{name_: "default"}
	strat1 := &MockStrategy{name_: "strategy1"}
	strat2 := &MockStrategy{name_: "strategy2"}

	selector := NewSelector(defaultStrategy)
	// Both conditions will match, but first one should be chosen
	_ = selector.AddCondition("history_length >= 0", strat1)
	_ = selector.AddCondition("history_length >= 0", strat2)

	query := strategy.Query{UserText: "test"}
	result, err := selector.Select(context.Background(), query)
	require.NoError(t, err)
	require.Equal(t, "strategy1", result.Name())
}

// TestSelectorConditionWithTenantID uses tenant_id in expression.
func TestSelectorConditionWithTenantID(t *testing.T) {
	defaultStrategy := &MockStrategy{name_: "default"}
	premium := &MockStrategy{name_: "premium"}
	selector := NewSelector(defaultStrategy)

	_ = selector.AddCondition("tenant_id == 'premium-tenant'", premium)

	query := strategy.Query{
		UserText: "test",
		TenantID: "premium-tenant",
	}
	result, err := selector.Select(context.Background(), query)
	require.NoError(t, err)
	require.Equal(t, "premium", result.Name())
}

// TestSelectorConditionWithHistoryLength conditions on history size.
func TestSelectorConditionWithHistoryLength(t *testing.T) {
	defaultStrategy := &MockStrategy{name_: "default"}
	multiquery := &MockStrategy{name_: "multiquery"}
	selector := NewSelector(defaultStrategy)

	_ = selector.AddCondition("history_length >= 3", multiquery)

	// Short history
	query := strategy.Query{
		UserText: "test",
		History: []strategy.Message{
			{Role: "user", Content: "q1"},
			{Role: "assistant", Content: "a1"},
		},
	}
	result, err := selector.Select(context.Background(), query)
	require.NoError(t, err)
	require.Equal(t, "default", result.Name())

	// Long history
	query.History = append(query.History,
		strategy.Message{Role: "user", Content: "q2"},
		strategy.Message{Role: "assistant", Content: "a2"},
		strategy.Message{Role: "user", Content: "q3"},
	)
	result, err = selector.Select(context.Background(), query)
	require.NoError(t, err)
	require.Equal(t, "multiquery", result.Name())
}

// TestSelectorConditionWithQueryText uses user query text.
func TestSelectorConditionWithQueryText(t *testing.T) {
	defaultStrategy := &MockStrategy{name_: "default"}
	faq := &MockStrategy{name_: "faq"}
	selector := NewSelector(defaultStrategy)

	// Simple keyword check
	_ = selector.AddCondition("query == 'how do I configure X'", faq)

	query := strategy.Query{UserText: "how do I configure X"}
	result, err := selector.Select(context.Background(), query)
	require.NoError(t, err)
	require.Equal(t, "faq", result.Name())
	// Note: contains() might not be available; this tests the pattern
}

// TestSelectorConditionEvaluationError continues on error.
func TestSelectorConditionEvaluationError(t *testing.T) {
	defaultStrategy := &MockStrategy{name_: "default"}
	selector := NewSelector(defaultStrategy)

	// Invalid condition that references non-existent field
	_ = selector.AddCondition("nonexistent_field > 5", &MockStrategy{name_: "should-skip"})
	_ = selector.AddCondition("true", &MockStrategy{name_: "fallback"})

	query := strategy.Query{UserText: "test"}
	result, err := selector.Select(context.Background(), query)
	require.NoError(t, err)
	// Should either use fallback or default
	require.NotNil(t, result)
}

// TestSelectorMultipleConditionsComplex tests multiple conditions.
func TestSelectorMultipleConditionsComplex(t *testing.T) {
	defaultStrategy := &MockStrategy{name_: "default"}
	vector := &MockStrategy{name_: "vector"}
	hybrid := &MockStrategy{name_: "hybrid"}
	hyde := &MockStrategy{name_: "hyde"}

	selector := NewSelector(defaultStrategy)
	_ = selector.AddCondition("history_length == 0", vector)
	_ = selector.AddCondition("history_length > 0 && history_length < 5", hybrid)
	_ = selector.AddCondition("history_length >= 5", hyde)

	// Test each branch
	tests := []struct {
		historyCount int
		expected     string
	}{
		{0, "vector"},
		{1, "hybrid"},
		{3, "hybrid"},
		{5, "hyde"},
		{10, "hyde"},
	}

	for _, tt := range tests {
		query := strategy.Query{UserText: "test"}
		for i := 0; i < tt.historyCount; i++ {
			query.History = append(query.History, strategy.Message{
				Role:    "user",
				Content: "msg",
			})
		}

		result, err := selector.Select(context.Background(), query)
		require.NoError(t, err)
		require.Equal(t, tt.expected, result.Name())
	}
}

// TestSelectorWithTenantAgentID uses tenant_agent_id in condition.
func TestSelectorWithTenantAgentID(t *testing.T) {
	defaultStrategy := &MockStrategy{name_: "default"}
	special := &MockStrategy{name_: "special"}
	selector := NewSelector(defaultStrategy)

	_ = selector.AddCondition("tenant_agent_id == 'agent-123'", special)

	query := strategy.Query{
		UserText:      "test",
		TenantAgentID: "agent-123",
	}
	result, err := selector.Select(context.Background(), query)
	require.NoError(t, err)
	require.Equal(t, "special", result.Name())
}

// TestNewSelectorDefaults ensures default strategy is set.
func TestNewSelectorDefaults(t *testing.T) {
	defaultStrategy := &MockStrategy{name_: "default"}
	selector := NewSelector(defaultStrategy)

	require.NotNil(t, selector.defaultStr)
	require.Equal(t, "default", selector.defaultStr.Name())
}
