args = RUST_BACKTRACE=1

run:
	$(args) cargo run run
pretrain:
	$(args) time cargo run pretrain
pretrain_encoder_decoder:
	$(args) cargo run pretrain_encoder_decoder
test:
	$(args) cargo test attention -- --nocapture --test-threads 1
profile:
	cargo flamegraph --root -- pretrain
