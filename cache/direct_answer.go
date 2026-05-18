package cache

import (
	"context"
	"sync"
	"time"

	"github.com/redis/go-redis/v9"
)

// DirectAnswerCache stores direct-answer decisions/results for repeated short queries.
type DirectAnswerCache interface {
	Get(ctx context.Context, key string) (string, bool, error)
	Set(ctx context.Context, key, value string, ttl time.Duration) error
}

type cacheEntry struct {
	value     string
	expiresAt time.Time
}

// InMemoryDirectAnswerCache is a small TTL cache used as L1.
type InMemoryDirectAnswerCache struct {
	mu         sync.RWMutex
	items      map[string]cacheEntry
	maxEntries int
}

func NewInMemoryDirectAnswerCache(maxEntries int) *InMemoryDirectAnswerCache {
	if maxEntries <= 0 {
		maxEntries = 1000
	}
	return &InMemoryDirectAnswerCache{
		items:      make(map[string]cacheEntry),
		maxEntries: maxEntries,
	}
}

func (c *InMemoryDirectAnswerCache) Get(_ context.Context, key string) (string, bool, error) {
	now := time.Now()
	c.mu.RLock()
	entry, ok := c.items[key]
	c.mu.RUnlock()
	if !ok {
		return "", false, nil
	}
	if now.After(entry.expiresAt) {
		c.mu.Lock()
		delete(c.items, key)
		c.mu.Unlock()
		return "", false, nil
	}
	return entry.value, true, nil
}

func (c *InMemoryDirectAnswerCache) Set(_ context.Context, key, value string, ttl time.Duration) error {
	if ttl <= 0 {
		ttl = 24 * time.Hour
	}
	now := time.Now()
	c.mu.Lock()
	if len(c.items) >= c.maxEntries {
		for k, v := range c.items {
			if now.After(v.expiresAt) {
				delete(c.items, k)
			}
		}
		if len(c.items) >= c.maxEntries {
			for k := range c.items {
				delete(c.items, k)
				break
			}
		}
	}
	c.items[key] = cacheEntry{
		value:     value,
		expiresAt: now.Add(ttl),
	}
	c.mu.Unlock()
	return nil
}

// RedisDirectAnswerCache is an optional L2 cache.
type RedisDirectAnswerCache struct {
	client *redis.Client
	prefix string
}

func NewRedisDirectAnswerCache(client *redis.Client, prefix string) *RedisDirectAnswerCache {
	if prefix == "" {
		prefix = "rag:direct_answer:"
	}
	return &RedisDirectAnswerCache{
		client: client,
		prefix: prefix,
	}
}

func (c *RedisDirectAnswerCache) Get(ctx context.Context, key string) (string, bool, error) {
	val, err := c.client.Get(ctx, c.prefix+key).Result()
	if err == redis.Nil {
		return "", false, nil
	}
	if err != nil {
		return "", false, err
	}
	return val, true, nil
}

func (c *RedisDirectAnswerCache) Set(ctx context.Context, key, value string, ttl time.Duration) error {
	if ttl <= 0 {
		ttl = 24 * time.Hour
	}
	return c.client.Set(ctx, c.prefix+key, value, ttl).Err()
}

// Tier returns which layer answered a Get call. Used purely for telemetry.
type Tier string

const (
	TierMiss Tier = ""
	TierL1   Tier = "l1"
	TierL2   Tier = "l2"
)

// TieredDirectAnswerCache combines L1 and optional L2 cache.
type TieredDirectAnswerCache struct {
	l1  DirectAnswerCache
	l2  DirectAnswerCache
	ttl time.Duration
}

func NewTieredDirectAnswerCache(l1, l2 DirectAnswerCache, ttl time.Duration) *TieredDirectAnswerCache {
	if ttl <= 0 {
		ttl = 24 * time.Hour
	}
	return &TieredDirectAnswerCache{
		l1:  l1,
		l2:  l2,
		ttl: ttl,
	}
}

func (c *TieredDirectAnswerCache) Get(ctx context.Context, key string) (string, bool, error) {
	if c.l1 != nil {
		if val, ok, err := c.l1.Get(ctx, key); err == nil && ok {
			return val, true, nil
		}
	}
	if c.l2 != nil {
		val, ok, err := c.l2.Get(ctx, key)
		if err != nil || !ok {
			return "", false, err
		}
		if c.l1 != nil {
			_ = c.l1.Set(ctx, key, val, c.ttl)
		}
		return val, true, nil
	}
	return "", false, nil
}

// GetWithTier reports which cache layer satisfied the request.
func (c *TieredDirectAnswerCache) GetWithTier(ctx context.Context, key string) (string, Tier, error) {
	if c.l1 != nil {
		if val, ok, err := c.l1.Get(ctx, key); err == nil && ok {
			return val, TierL1, nil
		}
	}
	if c.l2 != nil {
		val, ok, err := c.l2.Get(ctx, key)
		if err != nil || !ok {
			return "", TierMiss, err
		}
		if c.l1 != nil {
			_ = c.l1.Set(ctx, key, val, c.ttl)
		}
		return val, TierL2, nil
	}
	return "", TierMiss, nil
}

func (c *TieredDirectAnswerCache) Set(ctx context.Context, key, value string, ttl time.Duration) error {
	if ttl <= 0 {
		ttl = c.ttl
	}
	var lastErr error
	if c.l1 != nil {
		if err := c.l1.Set(ctx, key, value, ttl); err != nil {
			lastErr = err
		}
	}
	if c.l2 != nil {
		if err := c.l2.Set(ctx, key, value, ttl); err != nil {
			lastErr = err
		}
	}
	return lastErr
}
