args = RUST_BACKTRACE=1

run:
	$(args) cargo run --release run
train:
	$(args) nohup time cargo run --release train 2>&1 | tee train_log.log
merge:
	$(time cargo run --release merge 2>&1
print_stats:
	$(args) cargo run --release print_stats
test:
	$(args) cargo test attention -- --nocapture --test-threads 1
profile:
	CARGO_PROFILE_RELEASE_DEBUG=true cargo flamegraph --root -- train
dump_loss:
	cat train_log.log  | grep Loss | sed "s/Epoch    //g" | sed "s/\/.* Loss = /\t/g"
test_model:
	$(args) cargo run --release test_all
results:
	$(args) cargo run --release print_results
