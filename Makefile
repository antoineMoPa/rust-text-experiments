args = RUST_BACKTRACE=1

run:
	$(args) cargo run run
train_new:
	$(args) nohup time cargo run train_new 2>&1 > train_log.log &
	tail -f train_log.log
train:
	$(args) nohup time cargo run train 2>&1 > train_log.log &
	tail -f train_log.log
merge:
	$(time cargo run merge 2>&1
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
	CARGO_PROFILE_RELEASE_DEBUG=true cargo flamegraph --root -- train
dump_loss:
	cat train_log.log  | grep Loss | sed "s/Epoch    //g" | sed "s/\/.* Loss = /\t/g"
minitest_qa:
	cargo run run The squirrels twinkled at night. Q: What twinkled at night? A:
	cargo run run The squirrels climbed the tree Q: What climbed the tree? A:
	cargo run run The birds climbed the tree Q: What climbed the tree? A:
	cargo run run The squirrels twinkled at night. Q: What twinkled at night? A:
	cargo run run The boy kicked the ball. Q: Who kicked the ball? A:
	cargo run run The dog kicked the ball. Q: Who kicked the ball? A:
	cargo run run The car was red. Q: What was red? A: 
	cargo run run The dog was red. Q: What was red? A: 
	cargo run run The cat was red. Q: What was red? A: 
