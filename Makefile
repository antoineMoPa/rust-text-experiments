args = RUST_BACKTRACE=1

run:
	$(args) cargo run --release run
train_new:
	$(args) nohup time cargo run --release train_new 2>&1 | tee train_log.log
train:
	$(args) nohup time cargo run --release train 2>&1 | tee train_log.log
merge:
	$(time cargo run --release merge 2>&1
print_stats:
	$(args) cargo run --release print_stats
pretrain_encoder_decoder:
	$(args) cargo run --release pretrain_encoder_decoder
test:
	$(args) cargo test attention -- --nocapture --test-threads 1
profile:
	CARGO_PROFILE_RELEASE_DEBUG=true cargo flamegraph --root -- train
dump_loss:
	cat train_log.log  | grep Loss | sed "s/Epoch    //g" | sed "s/\/.* Loss = /\t/g"
test_model:
	cargo run --release self_test > self_test.txt
	cargo run --release qa_test > qa_test.txt
	cat *_test.txt | grep "success rate"
