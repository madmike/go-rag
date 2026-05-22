package selector

import (
	"context"
	"fmt"
	"github.com/expr-lang/expr"
	"github.com/expr-lang/expr/vm"
	"github.com/madmike/go-rag/strategy"
)

// Condition defines a strategy to select if the eval string returns true.
type Condition struct {
	EvalString string
	Strategy   strategy.Strategy
	program    *vm.Program
}

// Selector evaluates the query against conditions to pick the right strategy.
type Selector struct {
	conditions []Condition
	defaultStr strategy.Strategy
}

// NewSelector creates a new strategy selector.
func NewSelector(defaultStr strategy.Strategy) *Selector {
	return &Selector{
		conditions: make([]Condition, 0),
		defaultStr: defaultStr,
	}
}

// AddCondition adds an expr condition to the selector.
func (s *Selector) AddCondition(evalString string, strat strategy.Strategy) error {
	program, err := expr.Compile(evalString, expr.Env(map[string]interface{}{
		"query":           "",
		"history_length":  0,
		"tenant_id":       "",
		"tenant_agent_id": "",
	}))
	if err != nil {
		return fmt.Errorf("failed to compile condition '%s': %w", evalString, err)
	}

	s.conditions = append(s.conditions, Condition{
		EvalString: evalString,
		Strategy:   strat,
		program:    program,
	})

	return nil
}

// Select returns the first strategy whose condition evaluates to true.
func (s *Selector) Select(ctx context.Context, q strategy.Query) (strategy.Strategy, error) {
	// Construct the environment for expr evaluation
	env := map[string]interface{}{
		"query":           q.UserText,
		"history_length":  len(q.History),
		"tenant_id":       q.TenantID,
		"tenant_agent_id": q.TenantAgentID,
	}

	for _, cond := range s.conditions {
		output, err := expr.Run(cond.program, env)
		if err != nil {
			// Log error but continue to next condition
			continue
		}

		if match, ok := output.(bool); ok && match {
			return cond.Strategy, nil
		}
	}

	// Fallback to default strategy
	return s.defaultStr, nil
}
