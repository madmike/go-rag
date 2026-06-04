module github.com/madmike/go-rag

go 1.25.5

require (
	github.com/alicebob/miniredis/v2 v2.38.0
	github.com/expr-lang/expr v1.17.8
	github.com/madmike/go-ai-providers v0.0.2
	github.com/madmike/go-infra v0.0.1
	github.com/madmike/go-pipeline v0.0.1
	github.com/madmike/go-rerank v0.0.1
	github.com/redis/go-redis/v9 v9.18.0
	github.com/stretchr/testify v1.11.1
)

require (
	github.com/cespare/xxhash/v2 v2.3.0 // indirect
	github.com/davecgh/go-spew v1.1.2-0.20180830191138-d8f796af33cc // indirect
	github.com/dgryski/go-rendezvous v0.0.0-20200823014737-9f7001d12a5f // indirect
	github.com/kaptinlin/jsonrepair v0.2.8 // indirect
	github.com/kr/text v0.2.0 // indirect
	github.com/mattn/go-colorable v0.1.14 // indirect
	github.com/mattn/go-isatty v0.0.20 // indirect
	github.com/pmezard/go-difflib v1.0.1-0.20181226105442-5d4384ee4fb2 // indirect
	github.com/rs/zerolog v1.35.0 // indirect
	github.com/tailscale/hujson v0.0.0-20250605163823-992244df8c5a // indirect
	github.com/yuin/gopher-lua v1.1.1 // indirect
	go.uber.org/atomic v1.11.0 // indirect
	golang.org/x/sys v0.43.0 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
)

replace github.com/madmike/go-infra => ../infra

replace github.com/madmike/go-ai-providers => ../providers

replace github.com/madmike/go-pipeline => ../pipeline

replace github.com/madmike/go-rerank => ../rerank
