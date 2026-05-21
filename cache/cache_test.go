package cache

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/redis/go-redis/v9"
	"github.com/stretchr/testify/require"
)

// TestInMemoryCacheGet retrieves cached value before expiry.
func TestInMemoryCacheGet(t *testing.T) {
	cache := NewInMemoryDirectAnswerCache(100)

	err := cache.Set(context.Background(), "key1", "answer1", 1*time.Hour)
	require.NoError(t, err)

	value, ok, err := cache.Get(context.Background(), "key1")
	require.NoError(t, err)
	require.True(t, ok)
	require.Equal(t, "answer1", value)
}

// TestInMemoryCacheGetMiss returns false for missing key.
func TestInMemoryCacheGetMiss(t *testing.T) {
	cache := NewInMemoryDirectAnswerCache(100)

	value, ok, err := cache.Get(context.Background(), "missing")
	require.NoError(t, err)
	require.False(t, ok)
	require.Equal(t, "", value)
}

// TestInMemoryCacheGetExpired deletes and misses expired entries.
func TestInMemoryCacheGetExpired(t *testing.T) {
	cache := NewInMemoryDirectAnswerCache(100)

	// Set with very short TTL
	err := cache.Set(context.Background(), "key1", "answer1", 1*time.Millisecond)
	require.NoError(t, err)

	// Wait for expiry
	time.Sleep(2 * time.Millisecond)

	value, ok, err := cache.Get(context.Background(), "key1")
	require.NoError(t, err)
	require.False(t, ok)
	require.Equal(t, "", value)
}

// TestInMemoryCacheSet stores with default TTL.
func TestInMemoryCacheSet(t *testing.T) {
	cache := NewInMemoryDirectAnswerCache(100)

	// Set without explicit TTL (should use default 24h)
	err := cache.Set(context.Background(), "key1", "value1", 0)
	require.NoError(t, err)

	value, ok, err := cache.Get(context.Background(), "key1")
	require.NoError(t, err)
	require.True(t, ok)
	require.Equal(t, "value1", value)
}

// TestInMemoryCacheMaxEntries evicts when capacity exceeded.
func TestInMemoryCacheMaxEntries(t *testing.T) {
	cache := NewInMemoryDirectAnswerCache(3)

	// Fill to capacity
	require.NoError(t, cache.Set(context.Background(), "k1", "v1", 1*time.Hour))
	require.NoError(t, cache.Set(context.Background(), "k2", "v2", 1*time.Hour))
	require.NoError(t, cache.Set(context.Background(), "k3", "v3", 1*time.Hour))

	// Add one more, should evict
	require.NoError(t, cache.Set(context.Background(), "k4", "v4", 1*time.Hour))

	// One of the first three should be gone (at least one evicted due to capacity)
	// We can't be sure which one, so just verify we have at most capacity items
	count := 0
	for _, key := range []string{"k1", "k2", "k3", "k4"} {
		if _, ok, err := cache.Get(context.Background(), key); err == nil && ok {
			count++
		}
	}
	require.LessOrEqual(t, count, 3)
}

// TestInMemoryCacheDefaultSize uses default size if not specified.
func TestInMemoryCacheDefaultSize(t *testing.T) {
	cache := NewInMemoryDirectAnswerCache(0) // Should use default 1000
	require.NotNil(t, cache)
	require.Equal(t, 1000, cache.maxEntries)
}

// TestInMemoryCacheNegativeSize uses default for negative size.
func TestInMemoryCacheNegativeSize(t *testing.T) {
	cache := NewInMemoryDirectAnswerCache(-1)
	require.Equal(t, 1000, cache.maxEntries)
}

// TestInMemoryCacheConcurrentAccess tests thread-safe reads.
func TestInMemoryCacheConcurrentAccess(t *testing.T) {
	cache := NewInMemoryDirectAnswerCache(1000)
	cache.Set(context.Background(), "shared", "answer", 1*time.Hour)

	results := make(chan bool, 10)

	for i := 0; i < 10; i++ {
		go func() {
			val, ok, err := cache.Get(context.Background(), "shared")
			results <- (err == nil && ok && val == "answer")
		}()
	}

	for i := 0; i < 10; i++ {
		require.True(t, <-results)
	}
}

// TestRedisCacheGet retrieves value from Redis.
func TestRedisCacheGet(t *testing.T) {
	mr := miniredis.NewMiniredis()
	require.NoError(t, mr.Start())
	defer mr.Close()

	client := redis.NewClient(&redis.Options{Addr: mr.Addr()})
	cache := NewRedisDirectAnswerCache(client, "test:")

	err := cache.Set(context.Background(), "key1", "answer1", 1*time.Hour)
	require.NoError(t, err)

	value, ok, err := cache.Get(context.Background(), "key1")
	require.NoError(t, err)
	require.True(t, ok)
	require.Equal(t, "answer1", value)
}

// TestRedisCacheGetMiss handles missing keys.
func TestRedisCacheGetMiss(t *testing.T) {
	mr := miniredis.NewMiniredis()
	require.NoError(t, mr.Start())
	defer mr.Close()

	client := redis.NewClient(&redis.Options{Addr: mr.Addr()})
	cache := NewRedisDirectAnswerCache(client, "test:")

	value, ok, err := cache.Get(context.Background(), "missing")
	require.NoError(t, err)
	require.False(t, ok)
	require.Equal(t, "", value)
}

// TestRedisCacheSetWithTTL respects TTL.
func TestRedisCacheSetWithTTL(t *testing.T) {
	mr := miniredis.NewMiniredis()
	require.NoError(t, mr.Start())
	defer mr.Close()

	client := redis.NewClient(&redis.Options{Addr: mr.Addr()})
	cache := NewRedisDirectAnswerCache(client, "test:")

	// Set with short TTL
	err := cache.Set(context.Background(), "key1", "value1", 1*time.Millisecond)
	require.NoError(t, err)

	// Advance time past expiry
	mr.FastForward(2 * time.Millisecond)

	value, ok, err := cache.Get(context.Background(), "key1")
	require.NoError(t, err)
	require.False(t, ok)
	require.Equal(t, "", value)
}

// TestRedisCacheDefaultTTL uses default when not specified.
func TestRedisCacheDefaultTTL(t *testing.T) {
	mr := miniredis.NewMiniredis()
	require.NoError(t, mr.Start())
	defer mr.Close()

	client := redis.NewClient(&redis.Options{Addr: mr.Addr()})
	cache := NewRedisDirectAnswerCache(client, "test:")

	// Set without TTL (should use 24h)
	err := cache.Set(context.Background(), "key1", "value1", 0)
	require.NoError(t, err)

	value, ok, err := cache.Get(context.Background(), "key1")
	require.NoError(t, err)
	require.True(t, ok)
	require.Equal(t, "value1", value)
}

// TestRedisCacheCustomPrefix uses custom prefix.
func TestRedisCacheCustomPrefix(t *testing.T) {
	mr := miniredis.NewMiniredis()
	require.NoError(t, mr.Start())
	defer mr.Close()

	client := redis.NewClient(&redis.Options{Addr: mr.Addr()})
	cache := NewRedisDirectAnswerCache(client, "custom:prefix:")

	err := cache.Set(context.Background(), "key1", "value1", 1*time.Hour)
	require.NoError(t, err)

	// Verify key is stored with prefix
	val, err := client.Get(context.Background(), "custom:prefix:key1").Result()
	require.NoError(t, err)
	require.Equal(t, "value1", val)
}

// TestRedisCacheDefaultPrefix uses default prefix when empty.
func TestRedisCacheDefaultPrefix(t *testing.T) {
	mr := miniredis.NewMiniredis()
	require.NoError(t, mr.Start())
	defer mr.Close()

	client := redis.NewClient(&redis.Options{Addr: mr.Addr()})
	cache := NewRedisDirectAnswerCache(client, "")

	require.Equal(t, "rag:direct_answer:", cache.prefix)
}

// TestTieredCacheL1Hit returns from L1.
func TestTieredCacheL1Hit(t *testing.T) {
	l1 := NewInMemoryDirectAnswerCache(100)
	l1.Set(context.Background(), "key1", "answer1", 1*time.Hour)

	tiered := NewTieredDirectAnswerCache(l1, nil, 1*time.Hour)

	value, ok, err := tiered.Get(context.Background(), "key1")
	require.NoError(t, err)
	require.True(t, ok)
	require.Equal(t, "answer1", value)
}

// TestTieredCacheL2Hit returns from L2 and populates L1.
func TestTieredCacheL2Hit(t *testing.T) {
	l1 := NewInMemoryDirectAnswerCache(100)
	l2 := NewInMemoryDirectAnswerCache(100)
	l2.Set(context.Background(), "key1", "answer1", 1*time.Hour)

	tiered := NewTieredDirectAnswerCache(l1, l2, 1*time.Hour)

	value, ok, err := tiered.Get(context.Background(), "key1")
	require.NoError(t, err)
	require.True(t, ok)
	require.Equal(t, "answer1", value)

	// Verify L1 was populated
	l1Value, l1Ok, err := l1.Get(context.Background(), "key1")
	require.NoError(t, err)
	require.True(t, l1Ok)
	require.Equal(t, "answer1", l1Value)
}

// TestTieredCacheMiss returns miss when not in any layer.
func TestTieredCacheMiss(t *testing.T) {
	l1 := NewInMemoryDirectAnswerCache(100)
	l2 := NewInMemoryDirectAnswerCache(100)

	tiered := NewTieredDirectAnswerCache(l1, l2, 1*time.Hour)

	value, ok, err := tiered.Get(context.Background(), "missing")
	require.NoError(t, err)
	require.False(t, ok)
	require.Equal(t, "", value)
}

// TestTieredCacheL1OnlyHit uses L1 only when L2 nil.
func TestTieredCacheL1OnlyHit(t *testing.T) {
	l1 := NewInMemoryDirectAnswerCache(100)
	l1.Set(context.Background(), "key1", "answer1", 1*time.Hour)

	tiered := NewTieredDirectAnswerCache(l1, nil, 1*time.Hour)

	value, ok, err := tiered.Get(context.Background(), "key1")
	require.NoError(t, err)
	require.True(t, ok)
	require.Equal(t, "answer1", value)
}

// TestTieredCacheSet writes to both layers.
func TestTieredCacheSet(t *testing.T) {
	l1 := NewInMemoryDirectAnswerCache(100)
	l2 := NewInMemoryDirectAnswerCache(100)

	tiered := NewTieredDirectAnswerCache(l1, l2, 1*time.Hour)

	err := tiered.Set(context.Background(), "key1", "answer1", 1*time.Hour)
	require.NoError(t, err)

	// Verify in L1
	l1Val, l1Ok, _ := l1.Get(context.Background(), "key1")
	require.True(t, l1Ok)
	require.Equal(t, "answer1", l1Val)

	// Verify in L2
	l2Val, l2Ok, _ := l2.Get(context.Background(), "key1")
	require.True(t, l2Ok)
	require.Equal(t, "answer1", l2Val)
}

// TestTieredCacheDefaultTTL uses default TTL for Set.
func TestTieredCacheDefaultTTL(t *testing.T) {
	l1 := NewInMemoryDirectAnswerCache(100)

	tiered := NewTieredDirectAnswerCache(l1, nil, 1*time.Hour)

	// Set with 0 TTL should use tiered cache default
	err := tiered.Set(context.Background(), "key1", "answer1", 0)
	require.NoError(t, err)

	value, ok, err := tiered.Get(context.Background(), "key1")
	require.NoError(t, err)
	require.True(t, ok)
	require.Equal(t, "answer1", value)
}

// TestTieredCacheGetWithTierL1.
func TestTieredCacheGetWithTierL1(t *testing.T) {
	l1 := NewInMemoryDirectAnswerCache(100)
	l1.Set(context.Background(), "key1", "answer1", 1*time.Hour)

	tiered := NewTieredDirectAnswerCache(l1, nil, 1*time.Hour)

	value, tier, err := tiered.GetWithTier(context.Background(), "key1")
	require.NoError(t, err)
	require.Equal(t, "answer1", value)
	require.Equal(t, TierL1, tier)
}

// TestTieredCacheGetWithTierL2.
func TestTieredCacheGetWithTierL2(t *testing.T) {
	l1 := NewInMemoryDirectAnswerCache(100)
	l2 := NewInMemoryDirectAnswerCache(100)
	l2.Set(context.Background(), "key1", "answer1", 1*time.Hour)

	tiered := NewTieredDirectAnswerCache(l1, l2, 1*time.Hour)

	value, tier, err := tiered.GetWithTier(context.Background(), "key1")
	require.NoError(t, err)
	require.Equal(t, "answer1", value)
	require.Equal(t, TierL2, tier)
}

// TestTieredCacheGetWithTierMiss.
func TestTieredCacheGetWithTierMiss(t *testing.T) {
	l1 := NewInMemoryDirectAnswerCache(100)
	l2 := NewInMemoryDirectAnswerCache(100)

	tiered := NewTieredDirectAnswerCache(l1, l2, 1*time.Hour)

	value, tier, err := tiered.GetWithTier(context.Background(), "missing")
	require.NoError(t, err)
	require.Equal(t, "", value)
	require.Equal(t, TierMiss, tier)
}

// TestTieredCacheNilL1.
func TestTieredCacheNilL1(t *testing.T) {
	l2 := NewInMemoryDirectAnswerCache(100)
	l2.Set(context.Background(), "key1", "answer1", 1*time.Hour)

	tiered := NewTieredDirectAnswerCache(nil, l2, 1*time.Hour)

	value, ok, err := tiered.Get(context.Background(), "key1")
	require.NoError(t, err)
	require.True(t, ok)
	require.Equal(t, "answer1", value)
}

// TestTieredCacheSetL1Error continues to L2.
func TestTieredCacheSetL1Error(t *testing.T) {
	// Create a mock that returns error
	l1 := &MockCache{setErr: errors.New("l1 error")}
	l2 := NewInMemoryDirectAnswerCache(100)

	tiered := NewTieredDirectAnswerCache(l1, l2, 1*time.Hour)

	err := tiered.Set(context.Background(), "key1", "answer1", 1*time.Hour)
	require.Error(t, err) // Last error returned

	// Verify L2 still has the value
	value, ok, err := l2.Get(context.Background(), "key1")
	require.NoError(t, err)
	require.True(t, ok)
	require.Equal(t, "answer1", value)
}

// MockCache for testing error scenarios.
type MockCache struct {
	getErr error
	setErr error
}

func (m *MockCache) Get(ctx context.Context, key string) (string, bool, error) {
	if m.getErr != nil {
		return "", false, m.getErr
	}
	return "", false, nil
}

func (m *MockCache) Set(ctx context.Context, key, value string, ttl time.Duration) error {
	return m.setErr
}
