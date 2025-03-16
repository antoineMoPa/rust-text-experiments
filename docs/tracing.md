# Tracing

This repo use [tracing](https://github.com/tokio-rs/tracing/) to help spot performance bottlenecks.

```
cargo install inferno
```

```
# flamegraph
cat tracing.folded | inferno-flamegraph > tracing-flamegraph.svg

# flamechart
cat tracing.folded | inferno-flamegraph --flamechart > tracing-flamechart.svg
```
