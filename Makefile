SHELL := /bin/bash
.SHELLFLAGS := -o pipefail -c

args = RUST_BACKTRACE=1
features ?=

run:
	$(args) cargo run --release run
train:
	mkdir -p data
	$(args) nohup time cargo run --release train 2>&1 | tee train_log.log
train-flash:
	mkdir -p data
	$(args) RUSTFLAGS="-C linker=gcc" nohup time cargo run --release --features flash-attn train 2>&1 | tee train_log.log
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
	$(args) cargo run --release $(if $(features),--features $(features),) test_all
qa_test:
	$(args) cargo run --release $(if $(features),--features $(features),) qa_test
json_test:
	$(args) cargo run --release $(if $(features),--features $(features),) json_test
results:
	$(args) cargo run --release $(if $(features),--features $(features),) print_results
sweep_lr:
	mkdir -p data
	$(args) nohup time cargo run --release sweep-lr 2>&1 | tee sweep_log.log
sweep_results:
	@(printf '%s\t%s\t%s\t%s\t%s\n' LR L2 L3 QA JSON; cat lr_sweep.log | jq -r '[.LR, .Self_Test_Score_L2, .Self_Test_Score_L3, .QA_Test_Score, .JSON_Test_Score] | @tsv')
epoch_stats:
	@(printf '%s\t%s\t%s\t%s\t%s\t%s\n' Epoch LR L2 L3 QA JSON; cat per_epoch_stats.log | jq -r '[.Epoch, .LR, .Self_Test_Score_L2, .Self_Test_Score_L3, .QA_Test_Score, .JSON_Test_Score] | @tsv')
param_count:
	$(args) cargo run --release param_count
clean:
	rm -f *.log
	rm -f *.json
	rm -f data/*.bpe
