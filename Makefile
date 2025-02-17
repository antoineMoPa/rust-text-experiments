run:
	cargo run run
pretrain:
	cargo run pretrain
pretrain_encoder_decoder:
	cargo run pretrain_encoder_decoder
test:
	cargo test attention -- --nocapture --test-threads 1
