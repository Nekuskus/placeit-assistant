use itertools::Itertools;
use kusprint::*;
use regex::Regex;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::io::{stdin, stdout, Write};
use std::num::NonZeroU32;
use std::ops::RangeInclusive;
use rand::distributions::{Distribution, Uniform};
use std::time::{Duration, Instant};
use rayon::prelude::*;

use std::fs::{OpenOptions, read_to_string};
use std::path::Path;

const PROMPT: &str = "$";

#[derive(Debug, Clone)]
enum Slot {
    Set(i32),
    Range(RangeInclusive<i32>),
}

#[derive(Debug)]
enum FindError {
    Frozen,
    GameAlreadyWon,
    GameOver,
    AlreadyPlaced,
    OutOfRange,
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
            return Err(GameAlreadyWon); // game already won, board is all set
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
                if (v == 0 && match self.slots[0] { Slot::Range(_) => true, _ => false }) || v >= self.slots.len() {
                    return Err(OutOfRange);
                } else {
                    return Err(GameOver);
                }
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

    fn fplace(&mut self, val: i32) -> Result<String, String> {
        match self.find_best_placement(val) {
            Ok(idx) => {
                let old = self.slots.clone();
                match self.place((idx) as usize, val) {
                    Ok(_) => Ok(String::from(format!(
                        "Placed {val} in {}: {:?}",
                        idx + 1,
                        old[idx]
                    ).as_str())),
                    Err(e) => match e {
                        PlaceItError::SlotTaken => {
                            Err(String::from(format!("ERR: Slot {} already set, not changing", idx)))
                        }
                        PlaceItError::GameOver => {
                            Err(String::from("ERR: Cannot generate slots: no space to create enough ranges, game over, consider saving this score to local history."))
                        }
                    },
                }
            },
            Err(e) => match e {
                FindError::GameOver | FindError::OutOfRange => {
                    Err(String::from("ERR: Cannot place value, game over, consider saving this score to local history."))
                }
                _ => {
                    Err(String::from("ERR: Error when finding best placement: {e:?}"))
                }
            }
        }
    }

    // No need to construct strings or return matchable error variants when benchmarking
    fn fplace_no_str(&mut self, val: i32) -> Result<(), ()> {
        match self.find_best_placement(val) {
            Ok(idx) => {
                match self.place((idx) as usize, val) {
                    Ok(_) => Ok(()),
                    Err(_) => Err(()),
                }
            },
            Err(_) => Err(())
        }
    }

    fn gen_slots(count: u32, mut range: RangeInclusive<i32>) -> Option<Vec<Slot>> {
        match count {
            0 => Some(vec![]),
            _ => {
                let total = range.start().abs_diff(*range.end()) + 1;
                let slot_length = total / count;
                let mut remainder = total % count;

                let mut v = vec![];

                // I could do this with math but I prefer iterators to do this for me in a not error-prone manner
                if slot_length == 1 {
                    for _ in 0..count {
                        // TODO: add distribution (middle!)
                        let n = range.next().unwrap();
                        v.push(Slot::Range(n..=n));
                    }
                } else {
                    for _ in 0..count - 1 {
                        let first = range.next().unwrap();
                        // TODO: distribute in the middle instead
                        if remainder > 0 {
                            range = range.dropping((slot_length + 1 - 2) as usize);
                            remainder -= 1;
                        } else {
                            range = range.dropping((slot_length - 2) as usize);
                        }
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
                        .dropping((skip_count) as usize)
                        .next()
                        .unwrap();
                    v.push(Slot::Range(first..=last));
                }

                Some(v)
            }
        }
    }

}

fn flush_stdout() {
    stdout()
        .flush()
        .expect_err("ERR: Error flushing terminal")
        .unwrap();
}

fn read_file(filename: &Path) -> String {
    read_to_string(filename) 
        .unwrap()  // panic on possible file-reading errors
        .lines()  // split the string into an iterator of string slices
        .map(|s| s.trim().to_owned())
        .collect()  // gather them together into a vector
}

fn push_to_history(game: &PlaceIt, history: &mut HashMap<u32, (u32, Vec<i32>)>) {
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

    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open("history.ron").unwrap();
    ron::ser::to_writer(file, &history).unwrap();
}

#[allow(non_snake_case)]
mod combinatorics {
    use core::ops::RangeInclusive;

    pub struct PermuteSlots {
        state: Option<Vec<i32>>,
        count: usize,
        start: i32,
        end: i32
    }

    impl PermuteSlots {
        pub fn new(count: usize, val_range: RangeInclusive<i32>) -> Self {
            return Self {
                state: None,
                count: count,
                start: *val_range.start(),
                end: *val_range.end()
            }
        }

        pub fn count(&self) -> usize {
            super::combinatorics::V(self.start.abs_diff(self.end), self.count as u32) as usize
        }
    }

    impl Iterator for PermuteSlots {
        type Item = Vec<i32>;

        fn next(&mut self) -> Option<Self::Item> {
            if self.state.is_none() {
                self.state = Some(vec![self.start].repeat(self.count as usize));
                return self.state.clone();
            }

            let s = self.state.as_deref_mut().unwrap();

            if s.iter().all(|&v| v == self.end) {
                return None
            }
            
            for i in 0..s.len() {
                if s[i] < self.end {
                    s[i] += 1;
                    break;
                } else if s[i] > self.end { // overflowed by prev iter
                    s[i] = self.start;
                    s[i+1] += 1;
                }
            }
            
            return self.state.clone();
        }
    }

    pub const fn P(n: u32) -> u32 {
        let mut p = 1;
        let mut i = 2;
        while i <= n {
            p *= i;
            i += 1;
        }
        
        p
    }
    
    pub const fn V(n: u32, k: u32) -> u32 {
        P(n) / P(n-k)
    }
}

fn benchmark(count: u32, slots_count: NonZeroU32, val_range: RangeInclusive<i32>, history: &mut HashMap<u32, (u32, Vec<i32>)>) -> Duration {
    let s_count = slots_count.get();

    let between = Uniform::from(val_range.clone());

    let start = Instant::now();

    let results = (0..count).into_par_iter().map(|_| {
        let mut game = PlaceIt::new(PlaceIt::gen_slots(s_count, val_range.clone()).unwrap());

        let mut rng = rand::thread_rng();
        let mut used = vec![];
        for _ in 0..s_count {
            let mut num = between.sample(&mut rng);
            while used.contains(&num) {
                num = between.sample(&mut rng);
            }
            
            used.push(num);

            match game.fplace_no_str(num) {
                Err(_) => {
                    break;
                }
                _ => ()
            }
        }
        
        return game
            .slots
            .iter()
            .filter_map(|slot| match slot {
                Slot::Set(val) => Some(val),
                Slot::Range(_) => None,
            }).count()

    }).collect::<Vec<usize>>();
    
    let elapsed = start.elapsed();
    for res in results {
        history.entry(res as u32).or_insert((0, vec![])).0 += 1;
    }
    

    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open("history.ron").unwrap();
    ron::ser::to_writer(file, &history).unwrap();

    return elapsed;

}

#[allow(unused)]
fn analyze(slots_count: NonZeroU32, val_range: RangeInclusive<i32>) {
    use combinatorics::PermuteSlots;

    let count = slots_count.get();
    let mut game = PlaceIt::new(PlaceIt::gen_slots(count, val_range.clone()).unwrap());
    
    let permute = PermuteSlots::new(count as usize, val_range);

}

fn main() {
    let mut slots_count = NonZeroU32::new(20).unwrap();
    let mut val_range: RangeInclusive<i32> = 1..=999;

    let mut history: HashMap<u32, (u32, Vec<i32>)> = HashMap::new();

    if Path::new("history.ron").exists() {
        history = ron::from_str(read_file(Path::new("history.ron")).as_str()).unwrap();
    }

    let mut game = PlaceIt::new(PlaceIt::gen_slots(slots_count.get(), val_range.clone()).unwrap());
    println!("PlaceIt-Assistant shell (h for help):");
    println!(
        "INFO: Starting new game of PlaceIt (slots: {}, range: {:?})",
        slots_count, val_range
    );
    expr_print!(PROMPT);
    flush_stdout();
    let mut line = String::from("");

    let place_regex = Regex::new(r"(?:p|place) \d+ (?:-)?\d+").unwrap();
    let find_regex = Regex::new(r"(?:f|find) \d+").unwrap();
    let fplace_regex = Regex::new(r"(?:F|fplace) (?:-)?\d+").unwrap();
    let slots_regex = Regex::new(r"(?:s|slots) \d+").unwrap();
    let range_regex = Regex::new(r"(?:r|range) (?:-)?\d+ (?:-)?\d+").unwrap();
    let bench_regex = Regex::new(r"(?:b|benchmark) (?:-)?\d+").unwrap();

    loop {
        stdin()
            .read_line(&mut line)
            .expect_err("ERR: Error while reading:");
        let c = line.as_str().trim();
        match c {
            "h" | "help" => {
                println!("h (help)                   display this message");
                println!("p (place) <slot> <value>   place the number in a slot");
                println!("f (find) <value>           find the best slot for a number");
                println!("F (fplace) <value>         find the best slot for a number and places it there");
                println!("n (new)                    reset state, starting a new game and adding the current result to history");
                println!("H (history)                display history, in format <score>: <count> (percentage of game count) [list of last set nums for this score], skipping scores that were not yet achieved");
                println!("c (clear)                  clear history");
                println!("s (slots) <value>          set slots count (non-zero)");
                println!("r (range) <value> <value>  set value range");
                println!("l (list)                   list slots");
                println!("b (benchmark) <count>      todo - benchmark");
                println!("a (analyze)                todo - tests all possible games under current settings");
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
                                    eprintln!("ERR: Cannot generate slots: no space to create enough ranges. Game over, consider saving this score to local history.")
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
                                FindError::GameOver => {
                                    eprintln!("Placing {val} would cause a gameover.")
                                }
                                _ => {
                                    eprintln!("ERR: Error when finding best placement: {e:?}");
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
                        Ok(val) => {
                            match game.fplace(val) {
                                Ok(s) => println!("{}", s),
                                Err(s) => eprintln!("{}", s)
                            }
                        }
                        _ => {
                            eprintln!("ERR: Non-integer vals passed to place")
                        }
                    }
                }
            },
            "n" | "new" => {
                push_to_history(&game, &mut history);

                game = PlaceIt::new(PlaceIt::gen_slots(slots_count.get(), val_range.clone()).unwrap());
                println!(
                    "INFO: Starting new game of PlaceIt (slots: {}, range: {:?})",
                    slots_count, val_range
                );
            }
            "H" | "history" => {
                let list = history
                    .iter()
                    .sorted_by(|cur, other| return cur.0.cmp(&other.0))
                    .rev()
                    .collect::<Vec<_>>();
                let (total, sum_score) = history.iter().fold((0, 0), |acc, (score, (count, _))| return (acc.0 + count, acc.1 + score * count));
                for (score, (count, vals)) in list {
                    let percentage = *count as f32 / total as f32 * 100.0;
                    println!(
                        "{}: {} ({:.3}%) {:?}",
                        score,
                        count,
                        percentage,
                        vals.iter().sorted().collect::<Vec<_>>()
                    )
                }
                let average_score = sum_score as f32 / total as f32;
                println!("Average score: {}", average_score);
                println!("Count of games played: {}", total);
            }
            "c" | "clear" => {
                history = HashMap::new();

                let file = OpenOptions::new()
                    .create(true)
                    .write(true)
                    .truncate(true)
                    .open("history.ron").unwrap();
                ron::ser::to_writer(file, &history).unwrap();
            }
            _ if slots_regex.is_match(c) => {
                let split: Vec<_> = c.split(' ').collect();
                if split.len() != 2 {
                    eprintln!(
                        "ERR: Not enough params for slots (got {} required 2)",
                        split.len()
                    )
                } else {
                    match split[1].parse::<NonZeroU32>() {
                        Ok(val) => {
                            if val_range.start().abs_diff(*val_range.end()) / val > 1 {
                                slots_count = val;
                            }
                        }
                        _ => {
                            eprintln!("ERR: Non-integer vals passed to slots")
                        }
                    }
                }
            },
            _ if range_regex.is_match(c) => {
                let split: Vec<_> = c.split(' ').collect();
                if split.len() != 3 {
                    eprintln!(
                        "ERR: Not enough params for range (got {} required 3)",
                        split.len()
                    )
                } else {
                    let startres = split[1].parse::<i32>();
                    let endres = split[2].parse::<i32>();
                    match (startres, endres) {
                        (Ok(start), Ok(end)) => {
                            val_range = start..=end;
                        },
                        _ => {
                            eprintln!("ERR: Non-integer vals passed to range")
                        }
                    }
                }
            }
            _ if bench_regex.is_match(c) => {
                let split: Vec<_> = c.split(' ').collect();
                if split.len() != 2 {
                    eprintln!(
                        "ERR: Not enough benchmark for slots (got {} required 2)",
                        split.len()
                    )
                } else {
                    match split[1].parse::<u32>() {
                        Ok(val) => {
                            let length = benchmark(val, slots_count, val_range.clone(), &mut history).as_secs();
                            println!("Benchmark finished in {}s", length);
                        }
                        _ => {
                            eprintln!("ERR: Non-integer vals passed to benchmark")
                        }
                    }
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
            _ => { }
        }
        expr_print!(PROMPT);
        flush_stdout();
        line = String::from("");
    }
}
