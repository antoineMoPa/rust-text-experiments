"""
Generate fine-tuning pairs in { "result": ... } format.
Splits into train/val, ensuring no sample prompt matches json_test.txt exactly.
Output: finetune_train.json, finetune_val.json (NDJSON, one {"prompt","response"} per line)
"""

import json
import random

TEST_FILE = "../smoll-generated-corpus/level_5/json_test.txt"
TRAIN_FILE = "finetune_train.json"
VAL_FILE = "finetune_val.json"

TARGET_TOTAL = 2000
VAL_FRACTION = 0.15

FMT_NUM = 'Output in json, format { "result": number }'
FMT_BOOL = 'Output in json, format { "result": boolean }'

# ── load test prompts to exclude ──────────────────────────────────────────────
with open(TEST_FILE) as f:
    lines = [l.strip() for l in f if l.strip()]

test_prompts = {l for l in lines if l.startswith("[json]")}
print(f"Loaded {len(test_prompts)} test prompts to exclude.")

# ── generators ────────────────────────────────────────────────────────────────

def p(text, response):
    return "[json] " + text + " " + FMT_NUM, {"result": response}

def pb(text, response):
    return "[json] " + text + " " + FMT_BOOL, {"result": response}

def make_addition():
    a, b = random.randint(1, 200), random.randint(1, 200)
    return p(f"What is {a} + {b}?", a + b)

def make_subtraction():
    a = random.randint(1, 200)
    b = random.randint(0, a)
    return p(f"What is {a} - {b}?", a - b)

def make_multiplication():
    a, b = random.randint(2, 20), random.randint(2, 20)
    return p(f"What is {a} * {b}?", a * b)

def make_larger():
    a, b = random.sample(range(1, 100), 2)
    return p(f"Which is larger, {a} or {b}?", max(a, b))

def make_smaller():
    a, b = random.sample(range(1, 100), 2)
    return p(f"Which is smaller, {a} or {b}?", min(a, b))

def make_max():
    a, b = random.sample(range(1, 100), 2)
    return p(f"What is the maximum of {a} and {b}?", max(a, b))

def make_min():
    a, b = random.sample(range(1, 100), 2)
    return p(f"What is the minimum of {a} and {b}?", min(a, b))

def make_half():
    a = random.randrange(2, 200, 2)
    return p(f"What is half of {a}?", a // 2)

def make_double():
    a = random.randint(1, 100)
    return p(f"What is double {a}?", a * 2)

def make_store():
    n = random.randint(10, 200)
    s = random.randint(1, n)
    return p(f"A store has {n} items and sells {s}. How many remain?", n - s)

def make_hourly():
    r, h = random.randint(2, 20), random.randint(2, 10)
    return p(f"You earn {r} dollars per hour. How much do you earn in {h} hours?", r * h)

def make_days():
    d, w = random.randint(2, 10), random.randint(2, 8)
    return p(f"There are {d} days in a cycle. How many days are in {w} cycles?", d * w)

def make_movie():
    start, dur = random.randint(1, 20), random.randint(1, 5)
    return p(f"A movie starts at {start} and lasts {dur} hours. At what hour does it end?", start + dur)

def make_spending():
    total = random.randint(20, 200)
    spent = random.randint(1, total - 1)
    return p(f"You have {total} dollars and spend {spent}. How many dollars remain?", total - spent)

def make_driving():
    spd = random.choice([30, 40, 50, 60, 70, 80, 100])
    hrs = random.randint(1, 5)
    return p(f"A car travels at {spd} km per hour for {hrs} hours. How many km did it travel?", spd * hrs)

def make_reading():
    pages, days = random.randint(5, 50), random.randint(2, 10)
    return p(f"You read {pages} pages per day. How many pages in {days} days?", pages * days)

def make_grid():
    rows, cols = random.randint(2, 10), random.randint(2, 10)
    return p(f"A box has {rows} rows of {cols} items each. How many items total?", rows * cols)

def make_sensor():
    interval = random.choice([2, 3, 4, 5, 6, 10, 12, 15, 20])
    total = random.choice([60, 120, 180, 240, 300])
    return p(f"A sensor fires every {interval} seconds. How many times does it fire in {total} seconds?", total // interval)

def make_packs():
    items, packs = random.randint(2, 10), random.randint(2, 10)
    return p(f"Each pack has {items} items. How many items are in {packs} packs?", items * packs)

def make_marbles():
    a, b = random.randint(5, 50), random.randint(5, 50)
    return p(f"A jar has {a} red marbles and {b} blue marbles. How many marbles total?", a + b)

def make_cookies():
    total = random.randint(20, 100)
    given = random.randint(1, total - 1)
    return p(f"You bake {total} cookies and give away {given}. How many are left?", total - given)

def make_tickets():
    price, qty = random.randint(2, 20), random.randint(2, 10)
    return p(f"A ticket costs {price} dollars. How much do {qty} tickets cost?", price * qty)

def make_building():
    floors, apts = random.randint(2, 10), random.randint(2, 8)
    return p(f"A building has {floors} floors with {apts} apartments each. How many apartments total?", floors * apts)

def make_greater():
    a, b = random.randint(1, 100), random.randint(1, 100)
    return pb(f"Is {a} greater than {b}?", a > b)

def make_less():
    a, b = random.randint(1, 100), random.randint(1, 100)
    return pb(f"Is {a} less than {b}?", a < b)

def make_even():
    a = random.randint(1, 50)
    return pb(f"Is {a} even?", a % 2 == 0)

def make_odd():
    a = random.randint(1, 50)
    return pb(f"Is {a} odd?", a % 2 == 1)

def make_divisible():
    b = random.choice([2, 3, 4, 5, 6, 7, 8, 9, 10])
    a = random.randint(2, 50)
    return pb(f"Is {a} divisible by {b}?", a % b == 0)

def make_equal():
    a, b = random.randint(1, 30), random.randint(1, 30)
    return pb(f"Is {a} equal to {b}?", a == b)

FACTUAL_BOOL = [
    pb("Is a cat an animal?", True),
    pb("Is a rose a flower?", True),
    pb("Is silver a metal?", True),
    pb("Is a dolphin a mammal?", True),
    pb("Is wood a metal?", False),
    pb("Is a spider an insect?", False),
    pb("Is a frog a reptile?", False),
    pb("Is oxygen a gas?", True),
    pb("Is the sun a planet?", False),
    pb("Is the moon a satellite?", True),
    pb("Is fire cold?", False),
    pb("Is snow white?", True),
    pb("Is a hammer a tool?", True),
    pb("Is a guitar a percussion instrument?", False),
    pb("Is paper made from wood?", True),
    pb("Is a lemon sweet?", False),
    pb("Is a car a vehicle?", True),
    pb("Is milk a liquid?", True),
    pb("Is a cactus a plant?", True),
    pb("Is a brick soft?", False),
    pb("Is a penguin a bird?", True),
    pb("Is an oak a tree?", True),
    pb("Is a carrot a fruit?", False),
    pb("Is salt sweet?", False),
    pb("Is glass transparent?", True),
    pb("Is a submarine a boat?", True),
    pb("Is a bat a bird?", False),
    pb("Is a whale a mammal?", True),
    pb("Is iron a metal?", True),
    pb("Is a cloud made of water?", True),
]

NUMERIC_GENS = [
    make_addition, make_subtraction, make_multiplication,
    make_larger, make_smaller, make_max, make_min,
    make_half, make_double,
    make_store, make_hourly, make_days, make_movie, make_spending,
    make_driving, make_reading, make_grid, make_sensor, make_packs,
    make_marbles, make_cookies, make_tickets, make_building,
]

BOOL_GENS = [
    make_greater, make_less, make_even, make_odd, make_divisible, make_equal,
]

# ── generate ──────────────────────────────────────────────────────────────────
random.seed(42)

samples = []
seen = set(test_prompts)

def try_add(prompt, response):
    if prompt not in seen:
        seen.add(prompt)
        samples.append({"prompt": prompt, "response": response})
        return True
    return False

for prompt, response in FACTUAL_BOOL:
    try_add(prompt, response)

all_gens = NUMERIC_GENS * 3 + BOOL_GENS

attempts = 0
while len(samples) < TARGET_TOTAL and attempts < TARGET_TOTAL * 30:
    attempts += 1
    gen = random.choice(all_gens)
    try:
        prompt, response = gen()
        try_add(prompt, response)
    except Exception:
        continue

random.shuffle(samples)

val_size = int(len(samples) * VAL_FRACTION)
val = samples[:val_size]
train = samples[val_size:]

print(f"Total: {len(samples)}  |  train: {len(train)}  |  val: {len(val)}")

with open(TRAIN_FILE, "w") as f:
    for s in train:
        f.write(json.dumps(s) + "\n")

with open(VAL_FILE, "w") as f:
    for s in val:
        f.write(json.dumps(s) + "\n")

print(f"Written {TRAIN_FILE} and {VAL_FILE}")
