use itertools::Itertools;
use kusprint::*;
use regex::Regex;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::io::{stdin, stdout, Write};
use std::ops::RangeInclusive;

extern crate term_cursor as cursor;

const PROMPT: &str = "$";
static SLOTS_COUNT: u32 = 20;
static VAL_RANGE: RangeInclusive<i32> = 1..=999;

#[derive(Debug, Clone)]
enum Slot {
    Set(i32),
    Range(RangeInclusive<i32>),
}

#[derive(Debug)]
enum FindError {
    Frozen,
    GameWon,
    AlreadyPlaced,
    OutOfRange(usize),
}

enum PlaceItError {
    SlotTaken,
    GameOver,
}

struct PlaceIt {
    slots: Vec<Slot>,
    frozen: bool,
}

impl PlaceIt {
    fn new(slots: Vec<Slot>) -> Self {
        PlaceIt {
            slots: slots,
            frozen: false,
        }
    }

    fn find_best_placement(&self, val: i32) -> Result<usize, FindError> {
        use FindError::*;
        if self.frozen {
            return Err(Frozen);
        }

        if self.slots.iter().all(|s| match s {
            Slot::Set(_) => true,
            Slot::Range(_) => false,
        }) {
            return Err(GameWon); // game already won, board is all set
        } else if self.slots.iter().all(|s| match s {
            Slot::Set(set) => val == *set,
            Slot::Range(_) => false,
        }) {
            return Err(AlreadyPlaced); // value already on board
        }

        let res = self.slots.binary_search_by(|probe| match probe {
            Slot::Range(r) => {
                if r.contains(&val) {
                    return Ordering::Equal;
                } else if *r.end() < val {
                    return Ordering::Less;
                } else if *r.start() > val {
                    return Ordering::Greater;
                } else {
                    unreachable!()
                }
            }
            Slot::Set(set) => {
                return set.cmp(&val);
            }
        });

        match res {
            Ok(idx) => {
                return Ok(idx);
            }
            Err(v) => {
                return Err(OutOfRange(v));
            }
        }
    }

    fn place(&mut self, slot: usize, val: i32) -> Result<(), PlaceItError> {
        if self.frozen {
            return Err(PlaceItError::GameOver);
        }

        match self.slots[slot] {
            Slot::Set(_) => {
                return Err(PlaceItError::SlotTaken);
            }
            _ => (),
        };

        self.slots[slot] = Slot::Set(val);

        if slot > 0 {
            match &self.slots[slot - 1] {
                Slot::Range(r) => {
                    if val - 1 >= *r.start() {
                        self.slots[slot - 1] = Slot::Range(*r.start()..=val - 1);
                    } else {
                        self.slots[slot - 1] = Slot::Range(val - 1..=val - 1);
                    }
                }
                _ => (),
            };
        }

        if slot < self.slots.len() - 1 {
            match &self.slots[slot + 1] {
                Slot::Range(r) => {
                    if val + 1 <= *r.end() {
                        self.slots[slot + 1] = Slot::Range(val + 1..=*r.end());
                    } else {
                        self.slots[slot + 1] = Slot::Range(val + 1..=val + 1);
                    }
                }
                _ => (),
            };
        }

        let mut to_recalc: Vec<Vec<(usize, Slot)>> = self
            .slots
            .iter()
            .enumerate()
            .collect_vec()
            .split(|(_idx, s)| match s {
                Slot::Set(_) => true,
                Slot::Range(_) => false,
            })
            .filter(|s| s.len() != 0)
            .map(|s| {
                s.iter()
                    .map(|(idx, s)| (idx.clone(), s.to_owned().clone()))
                    .collect()
            })
            .collect();

        for x in to_recalc.iter_mut() {
            let (lower, upper, idx_lower, idx_upper) = match (x[0].clone(), x[x.len() - 1].clone())
            {
                ((x_lower, Slot::Range(first)), (x_upper, Slot::Range(last))) => (
                    *first.start().min(first.end()),
                    *last.start().max(last.end()),
                    x_lower,
                    x_upper,
                ),
                _ => unreachable!(),
            };

            let gen_iter =
                match PlaceIt::gen_slots(x.len() as u32, lower.min(upper)..=upper.max(lower)) {
                    Some(slots) => slots.into_iter().map(|s| s.to_owned().clone()),
                    None => {
                        self.frozen = true;
                        return Err(PlaceItError::GameOver);
                    }
                };
            *x = (idx_lower..=idx_upper).zip(gen_iter).collect();
        }

        let redistributed = to_recalc
            .iter()
            .flatten()
            .map(|tup| tup.to_owned())
            .collect::<Vec<(usize, Slot)>>();

        for (idx, v) in redistributed {
            self.slots[idx] = v;
        }

        return Ok(());
    }

    fn gen_slots(count: u32, mut range: RangeInclusive<i32>) -> Option<Vec<Slot>> {
        match count {
            0 => Some(vec![]),
            _ => {
                let total = range.start().abs_diff(*range.end()) + 1;
                let slot_length = total / count;
                let remainder = total % count;

                let mut v = vec![];

                // I could do this with math but I prefer iterators to do this for me in a not error-prone manner
                for _ in 0..count - 1 {
                    let first = range.next().unwrap();
                    range = range.dropping((slot_length - 2) as usize);
                    let last = range.next().unwrap();
                    v.push(Slot::Range(first..=last));
                }

                let first = range.next().unwrap();
                let skip_count = match slot_length.checked_sub(2) {
                    Some(val) => val,
                    None => {
                        return None;
                    }
                };
                let last = range
                    .dropping((skip_count + remainder) as usize)
                    .next()
                    .unwrap();
                v.push(Slot::Range(first..=last));

                Some(v)
            }
        }
    }
}

fn flush_stdout() {
    stdout()
        .flush()
        .inspect_err(|e| eprintln!("ERR: Error flushing terminal: {}", e))
        .unwrap();
}

fn main() {
    let mut history: HashMap<u32, (u32, Vec<i32>)> = HashMap::new();

    let mut game = PlaceIt::new(PlaceIt::gen_slots(SLOTS_COUNT, VAL_RANGE.clone()).unwrap());
    println!("PlaceIt-Assistant shell (h for help):");
    println!(
        "INFO: Starting new game of PlaceIt (slots: {}, range: {:?})",
        SLOTS_COUNT, VAL_RANGE
    );
    expr_print!(PROMPT);
    flush_stdout();
    let (twidth, _theight) = term_size::dimensions().unwrap();
    let mut line = String::from("");

    let place_regex = Regex::new(r"(?:p|place) \d+ \d+").unwrap();
    let find_regex = Regex::new(r"(?:f|find) \d+").unwrap();
    let fplace_regex = Regex::new(r"(?:F|fplace) \d+").unwrap();

    loop {
        stdin()
            .read_line(&mut line)
            .inspect_err(|e| eprintln!("ERR: Error while reading: {}", e))
            .unwrap();
        let c = line.as_str().trim();
        match c {
            "h" | "help" => {
                println!("h (help)                   display this message");
                println!("p (place) <slot> <value>   place the number in a slot");
                println!("f (find) <value>           find the best slot for a number");
                println!("F (fplace) <value>         find the best slot for a number and places it there");
                println!("n (new)                    reset state, starting a new game and adding the current result to history");
                println!("H (history)               display history, in format <score>: <count> [list of last set nums for this score], skipping scores that were not yet achieved");
                println!("s (slots)                  todo - set slots count");
                println!("r (range)                  todo - set value range");
                println!("l (list)                   list slots");
            }
            _ if place_regex.is_match(c) => {
                let split: Vec<_> = c.split(' ').collect();
                if split.len() != 3 {
                    eprintln!(
                        "ERR: Not enough params for place (got {} required 3)",
                        split.len()
                    )
                } else {
                    let idx_res = split[1].parse::<usize>();
                    let val_res = split[2].parse::<i32>();
                    match (idx_res, val_res) {
                        (Ok(idx), Ok(val)) => match game.place((idx - 1) as usize, val) {
                            Ok(_) => (),
                            Err(e) => match e {
                                PlaceItError::SlotTaken => {
                                    eprintln!("ERR: Slot {} already set, not changing", idx);
                                }
                                PlaceItError::GameOver => {
                                    eprintln!("ERR: Cannot generate slots: no space to create enough ranges, game over, consider saving this score to local history.")
                                }
                            },
                        },
                        _ => {
                            eprintln!("ERR: Non-integer vals passed to place")
                        }
                    }
                    if game.slots.iter().all(|s| match s {
                        Slot::Set(_) => true,
                        Slot::Range(_) => false,
                    }) {
                        eprintln!("INFO: The whole board is full, you win! Consider saving this score to local history.");
                    }
                }
            }
            _ if find_regex.is_match(c) => {
                let split: Vec<_> = c.split(' ').collect();
                if split.len() != 2 {
                    eprintln!(
                        "ERR: Not enough params for place (got {} required 2)",
                        split.len()
                    )
                } else {
                    match split[1].parse::<i32>() {
                        Ok(val) => match game.find_best_placement(val) {
                            Ok(idx) => {
                                println!(
                                    "Best slot for {val} is {}: {:?}",
                                    idx + 1,
                                    game.slots[idx]
                                );
                            }
                            Err(e) => match e {
                                FindError::OutOfRange(sug) => {
                                    eprintln!(
                                        "Error when finding best placement: {e:?} (suggests {sug})"
                                    );
                                }
                                _ => {
                                    eprintln!("Error when finding best placement: {e:?}");
                                }
                            },
                        },
                        _ => {
                            eprintln!("ERR: Non-integer vals passed to place")
                        }
                    }
                }
            }
            _ if fplace_regex.is_match(c) => {
                let split: Vec<_> = c.split(' ').collect();
                if split.len() != 2 {
                    eprintln!(
                        "ERR: Not enough params for place (got {} required 2)",
                        split.len()
                    )
                } else {
                    match split[1].parse::<i32>() {
                        Ok(val) => match game.find_best_placement(val) {
                            Ok(idx) => {
                                let old = game.slots.clone();
                                match game.place((idx) as usize, val) {
                                    Ok(_) => {
                                        println!(
                                            "Placed {val} in {}: {:?}",
                                            idx + 1,
                                            old[idx]
                                        );
                                    },
                                    Err(e) => match e {
                                        PlaceItError::SlotTaken => {
                                            eprintln!("ERR: Slot {} already set, not changing", idx);
                                        }
                                        PlaceItError::GameOver => {
                                            eprintln!("ERR: Cannot generate slots: no space to create enough ranges, game over, consider saving this score to local history.")
                                        }
                                    },
                                }
                            },
                            Err(e) => match e {
                                FindError::OutOfRange(sug) => {
                                    eprintln!(
                                        "Error when finding best placement: {e:?} (suggests {sug})"
                                    );
                                }
                                _ => {
                                    eprintln!("Error when finding best placement: {e:?}");
                                }
                            },
                        },
                        _ => {
                            eprintln!("ERR: Non-integer vals passed to place")
                        }
                    }
                }
            }
            "n" | "new" => {
                let set = game
                    .slots
                    .iter()
                    .filter_map(|slot| match slot {
                        Slot::Set(val) => Some(val),
                        Slot::Range(_) => None,
                    })
                    .copied()
                    .collect_vec();

                history.entry(set.len() as u32).or_insert((0, set)).0 += 1;

                game = PlaceIt::new(PlaceIt::gen_slots(SLOTS_COUNT, VAL_RANGE.clone()).unwrap());
                println!(
                    "INFO: Starting new game of PlaceIt (slots: {}, range: {:?})",
                    SLOTS_COUNT, VAL_RANGE
                );
            }
            "H" | "history" => {
                let list = history
                    .iter()
                    .sorted_by(|cur, other| return cur.0.cmp(&other.0))
                    .rev()
                    .collect::<Vec<_>>();
                for (score, (count, vals)) in list {
                    println!(
                        "{}: {} {:?}",
                        score,
                        count,
                        vals.iter().sorted().collect::<Vec<_>>()
                    )
                }
            }
            "l" | "list" => {
                println!(
                    "{:?}",
                    game.slots
                        .iter()
                        .enumerate()
                        .map(|(idx, slot)| (idx + 1, slot))
                        .collect::<Vec<_>>()
                )
            }
            "q" => {
                return;
            }
            _ => {
                expr_print!(
                    cursor::Up(1),
                    " ".repeat(twidth),
                    cursor::Left(twidth as i32 - 1)
                );
            }
        }
        expr_print!(PROMPT);
        flush_stdout();
        line = String::from("");
    }
}
