args = RUST_BACKTRACE=1

run:
	$(args) cargo run run
pretrain:
	$(args) time cargo run pretrain
print_stats:
	$(args) cargo run print_stats
print_stats_encoder_decoder:
	$(args) cargo run print_stats_encoder_decoder
print_dict_embeddings:
	$(args) cargo run print_dict_embeddings
pretrain_encoder_decoder:
	$(args) cargo run pretrain_encoder_decoder
test:
	$(args) cargo test attention -- --nocapture --test-threads 1
profile:
	cargo flamegraph --root -- pretrain
