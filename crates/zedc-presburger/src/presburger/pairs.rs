//! Which products of pairs of factor rows [`super::products`] forms (plan row C2-pairs-exact).
//!
//! Each product `p ⊗ q >= 0` of two linear rows `p >= 0`, `q >= 0` is read linearly, each
//! product atom `w * z` its own integer atom. An atom the formulas do not mention (no premise,
//! goal or definition holds it) is held only by product rows. When one product row is the only
//! live one holding such an atom with a nonzero coefficient, that row is dropped: whatever
//! point satisfies the other rows, the atom can be given a value of the sign of its coefficient
//! and large enough that the row holds too, since nothing else reads it. Dropped rows are
//! justified in the reverse of the order they go, each by the atom it alone held when it went,
//! so the verdict of the linear reading is the one all the products give, over the integers.
//! The rule is monotone (a row that can go can still go once others have gone), so the rows
//! left do not depend on the order of the tests or on names. (Proof: let rows r1, r2, ..
//! go in one order and let S be the rows another order leaves, from which no row can go. By
//! induction no r_k is in S: r_k could go from the rows left before it, all but
//! r1..r_{k-1}, a set that holds S, since none of those is in S; by monotonicity it could
//! then go from S, were it in S. So the first order removes no row the second keeps, and by
//! symmetry the two leave the same rows.)
//!
//! The coefficient of `w * z` in `p ⊗ q` is `p_w q_z + p_z q_w`, and `p_w q_w` for `w * w`. A
//! pair is tested from its factor rows without expanding the product: for each `w` of the
//! shorter row `q`, the `z` of the longer row `p` that do not give a free atom `w * z` are
//! collected, and the pair goes when they are fewer than the variables of `p`.

use std::cell::Cell;
use std::collections::{HashMap, HashSet};

use zed_graph::{Order, Worklist};
use zedc_int::int::Int;

use super::Refusal;
use super::formula::{Factor, Lin};

/// The steps [`kept`] may take for each unit of its input, and the terms and pairs
/// [`super::products`] may form for each unit of its own (the work meter of plan row
/// PRES-pairs-grid-reads). What it bounds: the reads, lookups and variables passed of
/// [`Index::frees`] and the pairs taken from the queue, against `k = R + T + P + M` (rows,
/// their terms, pairs, entries of `partners`); in [`super::products`], the pairs formed and
/// the terms of the products kept, against the terms of the rows multiplied. Past it the
/// question is refused ([`Refusal::Products`]); below it nothing changes: the meter only
/// stops the work, it never changes which pairs are kept, so every answer given is the one
/// the unmetered method gives. How it was chosen: the largest ratio measured with the meter
/// off on the tests of the crate (VM v5e1, job 45, 4,023 calls): steps of [`kept`] over its
/// input at most 35.1 (k = 37, a random case of
/// `tests::the_pairs_kept_are_those_left_by_expanding_every_product`; small inputs pay the
/// fixed cost of each pair's test), apart from the grid family at m = 16 (37.3, k = 976,
/// growing about as k^0.9 over the ratio); the pairs formed over the input of
/// [`super::products`] at most 1.14, and the terms of the products kept at most 24.6 (the
/// grid at m = 16), apart from Θ(k²) families such as
/// `tests::a_sum_holding_the_goal_reads_few_pairs` (76 at k = 100, 601 at k = 800), which
/// are refused. On the other tests (VM hm4, job 47, meter off): `l01_38_presburger`
/// (193 calls, up to the run's end at FAIL-power-too-large-built-before-refusal) steps at
/// most 13.3 and product terms 6.5; the ladder benchmarks and `l01_38_omega_test` (130
/// calls) 13.6 and 3.2. 40 is the next round number above 37.3. Neon's single-pass design
/// (XR-presburger4, NEON.md) measured at most 11.1 on its model and allows 16; that value
/// applies to it, not to this method. Why a limit at all: triangle detection reduces
/// to [`kept`] in linear size (proof below), so no near-linear method is likely for every
/// input; the author's rule (input only, DESIGN sec:plan) is to refuse what provably passes
/// a near-linear budget rather than run it, as [`Refusal::Work`] does.
pub(super) const PER_INPUT: u64 = 40;

#[cfg(test)]
thread_local! {
    /// Tests only: what [`Index::frees`] read on this thread, by any of its three ways: each
    /// pair read, each lookup of the pairs of two rows ([`Index::pairs_of`], counted whether
    /// it finds a pair or not), and each row whose list of pairs is entered (counted whether
    /// the list is empty or not).
    pub(super) static READS: Cell<u64> = const { Cell::new(0) };
    /// Tests only: the entries made in the lists of the pairs each pair blocked ([`kept`]).
    pub(super) static WATCHED: Cell<u64> = const { Cell::new(0) };
    /// Tests only: the steps the last call of [`kept`] on this thread charged its meter.
    pub(super) static SPENT: Cell<u64> = const { Cell::new(0) };
    /// Tests only: the input `k` of the last call of [`kept`] on this thread ([`PER_INPUT`]).
    pub(super) static INPUT: Cell<u64> = const { Cell::new(0) };
    /// Tests only: `Some(n)` gives the call of [`kept`] after the next `n` on this thread a
    /// limit of zero steps, so it refuses; `None` leaves every call its limit.
    pub(super) static REFUSE_AFTER: Cell<Option<usize>> = const { Cell::new(None) };
    /// Tests only: the rows on the first factor of a wanted product that
    /// [`super::products`] walked to form pairs, on this thread.
    pub(super) static WALKED: Cell<u64> = const { Cell::new(0) };
}

/// Counts one row walked by [`super::products`] ([`WALKED`]), in tests.
#[cfg(test)]
pub(super) fn walked() {
    WALKED.with(|n| n.set(n.get() + 1));
}

/// Counts one read ([`READS`]), in tests; nothing otherwise.
fn read() {
    #[cfg(test)]
    READS.with(|n| n.set(n.get() + 1));
}

/// Enters the pair `e` in the list of the pair `f` that blocks one of its atoms, to test `e`
/// again when `f` goes; counted ([`WATCHED`]) in tests.
fn watch_on(watch: &mut [Vec<usize>], f: usize, e: usize) {
    #[cfg(test)]
    WATCHED.with(|n| n.set(n.get() + 1));
    watch[f].push(e);
}

/// Whether each of `pairs`, the unordered pairs of indices into `rows`, is kept: the pairs left
/// when every pair whose product alone holds an atom with no entry in `partners` (the factors
/// each factor is multiplied by in a product the formulas mention) is dropped, to a fixpoint.
///
/// Complexity tradeoff: the naive method expands every product, Σ |p| × |q| terms over the
/// pairs, Θ(k²) for one row of k factors paired with a row on each of them, such as one sum
/// `b0 + ... + b{k-1}` beside the products `b{i} * c{i}` (33.5 times the instructions at 8
/// times the links, C2-pairs-long-row), and then counts the holders of each atom again after
/// each round of drops, O(rounds × Σ |p| × |q|). Here no product is expanded. The test of a
/// pair `{p, q}`, `|q| <= |p|`, tries each `w` of `q`, those on the fewest pairs first, and
/// stops at the first that frees an atom; for one `w` it costs O(|q| × log) for the atoms that
/// cancel inside `p ⊗ q`, O(min(|p|, partners of w) × log) for those mentioned, and, for each
/// row `r` holding `w`, O(log P + min(|r|, |p|) × log) to look up a live pair `{r, p}` that
/// holds every `w * z` (when `p` does not hold `w`, or `r` and `p` share only `w`), and failing
/// that, for each live pair `{r, s}`, O(min(|s|, |p|) × log) for the variables `s` and `p`
/// share; or, when the variables of `p` not yet known to be blocked are on fewer
/// pairs than `w` is, for each such `z` O(log) for each live pair on a row holding `z`; or, for
/// each such `z`, O(log) for each row holding `w` and row holding `z`, the pair of the two
/// found by binary search (see the comment in [`Index::frees`]), whichever reads fewest
/// pairs. After its first test a pair is
/// tested again only when a pair that held one of its atoms at its last test is dropped, so
/// each drop costs a test of the pairs that
/// share an atom with it, and each pair is tested at most once more than the number of its
/// blockers that go. On the family the sum is only ever `p`, each `w` is on a constant number
/// of pairs, and the whole costs O(k log k); with one sum `x + z0 + ... + z{k-1}` and the row
/// `x - 1` beside k rows `y - w{j}`, the test of `{x - 1, y - w{j}}` reads the two pairs on
/// `w{j}`, not the k on `x`, and the whole is again O(k log k)
/// (`products_whose_premise_holds_k_atoms_in_no_product_scale`); and when each `w{j}` is also
/// on a sum `w0 + ... + w{k-1}` paired with a row for each product `w{j} * c{j}`, that test
/// looks up the one pair of `x - 1` with each row on `w{j}`, O(log k), not the k pairs of the
/// sum (`products_beside_a_sum_whose_atoms_have_products_scale`). In general it is not
/// O(n log n): many long rows that share variables cost O(|s| × |p|) for each pair read, and
/// none is likely. Triangle detection reduces to this function in linear size (XR-presburger4
/// round 2, L6): for a graph of n vertices and M edges, take the rows `u{a}` and
/// `S{c} = sum of u{d} over the neighbors d of c`, the pairs `{u{a}, S{c}}` for each ordered
/// edge `(a, c)` and `{u{a}, u{b}}` for each edge, and the mentioned atoms `u{a} * u{a}`.
/// Proof: an atom `u{a} * u{d}` of `{u{a}, S{c}}`, `d != a`, is also held by `{u{d}, S{c}}`
/// (all coefficients 1, so nothing cancels), so the first such pair to be dropped would hold no
/// atom alone, and none is; `{u{a}, u{b}}` holds only `u{a} * u{b}`, also held exactly by the
/// pairs `{u{a}, S{c}}` and `{u{b}, S{c}}` with `c` adjacent to both, so it is kept exactly
/// when the edge `ab` lies in a triangle. QED. Checked against a naive fixpoint on 201 graphs
/// (scratch program, not committed). The reduction gives all-edge triangle membership (for
/// each edge, whether it lies in a triangle), and so triangle detection too; it is proved for
/// `kept` on arbitrary rows and pairs, not through the pairs [`super::products`] forms. So an
/// O(k polylog k) method here would detect triangles in a graph of m edges in almost linear
/// time. That is a conditional bar, not a proof that no such method exists: it would refute
/// Conjecture 4 of Abboud and Vassilevska
/// Williams, "Popular conjectures imply strong lower bounds for dynamic problems" (FOCS 2014,
/// arXiv:1402.0054): "any algorithm requires m^{1+\delta-o(1)} time in expectation to detect
/// whether an m edge graph contains a triangle". That paper gives O(min(m^1.41, n^omega)) as
/// the fastest algorithm in 2014; Dumitrescu, "A Strongly Subcubic Combinatorial Algorithm for
/// Triangle Detection with Applications" (arXiv:2403.01085, 2024, a preprint), reports a
/// randomized algorithm "running in O(n^{7/3}) time, or alternatively in O(m^{4/3}) time" with
/// one-sided error at most 10^-30, still above near-linear. The lower bound the conjecture
/// states is for the worst case; it classifies no single input, and says nothing about a
/// restricted family such as the grid below (XR-presburger4 round 4, LIT4-A, Codex, from the
/// primary sources).
/// Searched for a near-linear test of the pairs that alone hold an atom under deletions
/// (paperclip, 8 searches of 227 to 245 results each, 2026-11-22): none found. Gemini's grid
/// family, as written, cannot reach this function (its parallel rows are one row each after
/// `Formula::and`); on Gemini's extraction the reads grew 19.62 times at 7.83 times the pairs
/// (about k^1.45, two points). Its variant with no two rows parallel does reach it: from
/// m = 16 to 32 the products made grew 4 times (512 to 2,048), the terms they hold 7.1 times
/// (5,328 to 37,792) and the reads 13.2 times (14,473 to 191,505, about k^1.86) when it ran
/// to the end, before the meter: near the square of the products, not within a constant of
/// the terms, past the scaling limit. Under the meter the family is refused in steps linear in
/// its input (`tests::the_meter_refuses_the_grid_family_in_steps_linear_in_its_input`);
/// whether a method decides it in O((products + terms) log) is plan row PRES-pairs-grid-reads.
/// Tests:
/// `tests::the_pairs_kept_are_those_left_by_expanding_every_product` (against the naive
/// method),
/// `l01_38_presburger::ladder::products_whose_factors_one_sum_on_the_factors_joins_scale`, and
/// for the meter `tests::the_meter_refuses_the_grid_family_in_steps_linear_in_its_input`.
///
/// `Err(Refusal::Products)` when the steps pass [`PER_INPUT`] times the input `k`: each step
/// charged is O(log k) work, the meter is checked before each read and at each pair taken from
/// the queue, and the work between two checks is O(k log k) (one call of [`Index::shared`] or
/// [`Index::cancels_nothing`], or one pass over a row's variables), so a refusal costs
/// O(PER_INPUT × k log k) and an answer the same at most.
pub(super) fn kept(
    rows: &[&Lin],
    pairs: &[(usize, usize)],
    partners: &HashMap<Factor, Vec<Factor>>,
) -> Result<Vec<bool>, Refusal> {
    let index = Index::new(rows, pairs, partners);
    let n = pairs.len();
    let mut live = vec![true; n];
    let mut queue = Worklist::all(n, Order::Fifo);
    // The pairs that each pair blocked at their last test, to test again when it goes.
    let mut watch: Vec<Vec<usize>> = vec![Vec::new(); n];
    let answer = loop {
        let Some(e) = queue.pop() else {
            break Ok(live);
        };
        index.charge(1);
        if !live[e] {
            continue;
        }
        // A test stopped by the meter may answer wrongly; its answer is never used.
        let frees = index.frees(e, &live, &mut watch);
        if index.over() {
            break Err(Refusal::Products);
        }
        if frees {
            live[e] = false;
            for f in std::mem::take(&mut watch[e]) {
                if live[f] {
                    queue.push(f);
                }
            }
        }
    };
    #[cfg(test)]
    SPENT.with(|s| s.set(index.spent.get()));
    #[cfg(test)]
    INPUT.with(|s| s.set(index.input));
    answer
}

/// The factor rows and pairs of [`kept`], with each factor numbered.
struct Index<'a> {
    /// Each row's variables, by number, ascending, with their coefficients.
    vars: Vec<Vec<(usize, &'a Int)>>,
    pairs: &'a [(usize, usize)],
    /// The rows holding each variable.
    holding: Vec<Vec<usize>>,
    /// Each row's partners in a pair, with the pair.
    opp: Vec<Vec<(usize, usize)>>,
    /// The number of pairs on the rows holding each variable: the order in which the variables
    /// of a row are tried.
    weight: Vec<usize>,
    /// Each pair as its two rows, the lesser first, with its place in `pairs`, in that order:
    /// the pairs of two given rows are found by binary search ([`Index::pairs_of`]).
    at: Vec<((usize, usize), usize)>,
    /// The variables each variable is multiplied by in a product the formulas mention,
    /// ascending.
    mentioned: Vec<Vec<usize>>,
    /// The steps charged so far ([`PER_INPUT`]).
    spent: Cell<u64>,
    /// Tests only: the input `k = R + T + P + M` (rows, terms, pairs, entries of `partners`).
    #[cfg(test)]
    input: u64,
    /// The steps allowed: [`PER_INPUT`] times the input `k = R + T + P + M`.
    limit: u64,
}

impl<'a> Index<'a> {
    fn new(
        rows: &[&'a Lin],
        pairs: &'a [(usize, usize)],
        partners: &HashMap<Factor, Vec<Factor>>,
    ) -> Index<'a> {
        let mut number: HashMap<Factor, usize> = HashMap::new();
        let mut holding: Vec<Vec<usize>> = Vec::new();
        let vars: Vec<Vec<(usize, &Int)>> = rows
            .iter()
            .enumerate()
            .map(|(r, row)| {
                let mut vs: Vec<(usize, &Int)> = row
                    .terms
                    .iter()
                    .filter_map(|(a, c)| {
                        let f = Factor::of(a)?;
                        let next = number.len();
                        let v = *number.entry(f).or_insert(next);
                        if v == holding.len() {
                            holding.push(Vec::new());
                        }
                        holding[v].push(r);
                        Some((v, c))
                    })
                    .collect();
                vs.sort_unstable_by_key(|&(v, _)| v);
                vs
            })
            .collect();
        let mut opp: Vec<Vec<(usize, usize)>> = vec![Vec::new(); rows.len()];
        for (e, &(p, q)) in pairs.iter().enumerate() {
            opp[p].push((q, e));
            if p != q {
                opp[q].push((p, e));
            }
        }
        // Each variable's rows are kept only when they are on a pair (XR-presburger4 round 2,
        // L1). Every use of `holding` looks for pairs on its rows, and a row on no pair has
        // none, so no answer changes: a pair `{r, s}` that holds `w * z` has a row holding `w`
        // and a row holding `z` (its coefficient is `r_w s_z + r_z s_w`, or `r_w s_w` for
        // `z = w`), and both rows are on that pair, so both stay. What changes is the cost:
        // with k rows `z + u{j}` on no pair beside k rows `y + j z` each paired with `x - 1`,
        // the test of each pair `{x - 1, y + j z}` walked the k rows on no pair among the rows
        // holding `z`, whose `weight` (the pairs on them, 0) did not count them: Θ(k²) for the k
        // pairs (Codex, round 2). Test `a_premise_of_rows_on_no_pair_reads_few_lists`.
        for rs in &mut holding {
            rs.retain(|&r| !opp[r].is_empty());
        }
        let weight = holding
            .iter()
            .map(|rs| rs.iter().map(|&r| opp[r].len()).sum())
            .collect();
        let mut at: Vec<((usize, usize), usize)> = (pairs.iter().enumerate())
            .map(|(e, &(p, q))| ((p.min(q), p.max(q)), e))
            .collect();
        at.sort_unstable();
        let mut mentioned: Vec<Vec<usize>> = vec![Vec::new(); holding.len()];
        for (u, ps) in partners {
            if let Some(&u) = number.get(u) {
                mentioned[u] = ps.iter().filter_map(|p| number.get(p).copied()).collect();
                mentioned[u].sort_unstable();
            }
        }
        let k = rows.len()
            + rows.iter().map(|r| r.terms.len()).sum::<usize>()
            + pairs.len()
            + partners.values().map(Vec::len).sum::<usize>();
        let input = k as u64;
        #[allow(unused_mut)]
        let mut limit = PER_INPUT.saturating_mul(input);
        #[cfg(test)]
        REFUSE_AFTER.with(|c| match c.get() {
            Some(0) => {
                limit = 0;
                c.set(None);
            }
            Some(n) => c.set(Some(n - 1)),
            None => {}
        });
        Index {
            vars,
            pairs,
            holding,
            opp,
            weight,
            at,
            mentioned,
            spent: Cell::new(0),
            #[cfg(test)]
            input,
            limit,
        }
    }

    /// Charges `n` steps to the meter ([`PER_INPUT`]).
    fn charge(&self, n: usize) {
        self.spent.set(self.spent.get().saturating_add(n as u64));
    }

    /// Whether the steps charged passed the limit.
    fn over(&self) -> bool {
        self.spent.get() > self.limit
    }

    /// One read: counted in tests ([`READS`]) and charged one step; whether the meter allows
    /// more.
    fn read(&self) -> bool {
        read();
        self.charge(1);
        !self.over()
    }

    /// The pairs of the rows `r` and `s`, by binary search in `at`: O(log P + found) for P pairs.
    /// Each call is counted as a read ([`READS`]), and so is each pair found; none are found
    /// once the meter is spent.
    fn pairs_of(&self, r: usize, s: usize) -> impl Iterator<Item = usize> + '_ {
        let key = (r.min(s), r.max(s));
        let from = if self.read() {
            self.at.partition_point(|&(k, _)| k < key)
        } else {
            self.at.len()
        };
        (self.at[from..].iter())
            .take_while(move |&&(k, _)| k == key)
            .map(|&(_, f)| f)
            .take_while(|_| self.read())
    }

    /// The coefficient of `w` in row `r`, zero when it has none.
    fn coef(&self, r: usize, w: usize) -> Option<&'a Int> {
        zed_collections::sorted::sorted_get(&self.vars[r], &w).copied()
    }

    /// Whether the coefficient of `w * z` in `r ⊗ s` is not zero.
    fn holds(&self, r: usize, s: usize, w: usize, z: usize) -> bool {
        let times = |a: usize, b: usize| match (self.coef(r, a), self.coef(s, b)) {
            (Some(x), Some(y)) => x * y,
            _ => Int::ZERO,
        };
        if w == z {
            !times(w, w).is_zero()
        } else {
            !(&times(w, z) + &times(z, w)).is_zero()
        }
    }

    /// The variables of `p` that `s` holds too, through the shorter of the two.
    fn shared(&self, s: usize, p: usize) -> Vec<usize> {
        let (short, long) = if self.vars[s].len() <= self.vars[p].len() {
            (s, p)
        } else {
            (p, s)
        };
        self.charge(self.vars[short].len());
        (self.vars[short].iter())
            .map(|&(v, _)| v)
            .filter(|&v| self.coef(long, v).is_some())
            .collect()
    }

    /// Whether no variable `z != w` that the rows `r` and `p` share cancels the atom `w * z` in
    /// `r ⊗ p` (`r_w p_z + r_z p_w != 0` for each), read through the shorter of the two:
    /// O(min(|r|, |p|) log). With `r` holding `w`, it says that `r ⊗ p` holds `w * z` for
    /// every `z` of `p` (proof at its use in [`Index::frees`]).
    fn cancels_nothing(&self, r: usize, p: usize, w: usize) -> bool {
        let (short, long) = if self.vars[r].len() <= self.vars[p].len() {
            (r, p)
        } else {
            (p, r)
        };
        self.charge(self.vars[short].len());
        (self.vars[short].iter())
            .all(|&(v, _)| v == w || self.coef(long, v).is_none() || self.holds(r, p, w, v))
    }

    /// Whether the live pair `e` alone holds an atom `w * z` that the formulas do not mention,
    /// against the live pairs `live`; when it does not, `e` is entered in `watch` under each
    /// pair found to hold one of its atoms.
    fn frees(&self, e: usize, live: &[bool], watch: &mut [Vec<usize>]) -> bool {
        let (a, b) = self.pairs[e];
        let (q, p) = if self.vars[a].len() <= self.vars[b].len() {
            (a, b)
        } else {
            (b, a)
        };
        let width = self.vars[p].len();
        let mut ws: Vec<usize> = self.vars[q].iter().map(|&(w, _)| w).collect();
        self.charge(ws.len());
        ws.sort_by_key(|&w| (self.weight[w], w));
        'w: for w in ws {
            // A spent meter ends the test; [`kept`] refuses and never reads its answer.
            if self.over() {
                return true;
            }
            // The `z` of `p` for which `w * z` is not free.
            let mut not: HashSet<usize> = HashSet::new();
            let ms = &self.mentioned[w];
            self.charge(ms.len().min(width) + self.vars[q].len());
            if ms.len() <= width {
                not.extend(ms.iter().filter(|&&z| self.coef(p, z).is_some()));
            } else {
                not.extend(self.vars[p].iter().map(|&(z, _)| z));
                not.retain(|z| ms.binary_search(z).is_ok());
            }
            // Only for `z` held by `q` too can the coefficient in `e` be zero.
            not.extend(
                (self.vars[q].iter())
                    .map(|&(z, _)| z)
                    .filter(|&z| self.coef(p, z).is_some() && !self.holds(p, q, w, z)),
            );
            if not.len() == width {
                continue 'w;
            }
            // Three ways to find the live pairs that hold an atom `w * z`
            // (PRES-growth-ladder-bytes): from the rows holding `w` (last, `weight[w]` pairs,
            // each read for the variables it shares with `p`); from the rows holding each `z`
            // of `p` still open (`by_z`, `weight[z]` pairs each); or, for each open `z`, by
            // looking up the pairs of each row holding `w` with each row holding `z`
            // (`by_rows`, `|holding[w]| × |holding[z]|` binary searches in `at`). All three
            // are complete: a pair `{r, s}` holds `w * z` only when its coefficient
            // `r_w s_z + r_z s_w` (`r_w s_w` for `z = w`) is not zero, so one of its rows holds
            // `w` and one holds `z`, and each way reads every live pair with that property
            // (the third reads exactly those pairs; the coefficient is the same with `r` and
            // `s` swapped). So all put the same `z` into `not`, and `e` frees an atom through
            // `w` by one exactly when it does by another; for each `z` put in through a pair,
            // one blocking pair is watched, which is all `kept` needs (a pair is tested again
            // when a pair that blocked one of its atoms goes; while the watched blocker of `z`
            // stays, `z` stays blocked). The way taken is the one that reads fewest pairs.
            // Their costs, for P pairs: from `w`, `weight[w]` pairs, each O(log), and for each
            // live pair `{r, s}` read, O(min(|s|, |p|) × log) for the variables `s` shares with
            // `p` (`shared`); from the open `z`, the sum over them of `weight[z]` pairs, each
            // O(log), and one entry into the list of each row holding such a `z`; by rows, the
            // sum over them of `|holding[w]| × |holding[z]|` lookups, each O(log P). The
            // estimate compares `weight[w]` with the two sums, summed only until both pass
            // `weight[w]` (each term is at least 1, as `z` is on `p`, which is on `e`, and `w`
            // is on `q`; a row entered has a pair, as `holding` keeps only rows on a pair), in
            // saturating arithmetic, so that a sum past a word cannot wrap to a small one.
            // Choosing costs O(|not| + min(weight[w], open z)):
            // the scan of `p`'s variables passes over the `z` already in `not` on its way to
            // the open ones (XR-presburger4 round 1 B5), and each of those was put into `not`
            // above, so the building of `not` pays for passing it. From the open `z`, the
            // first `z` that no live pair blocks frees an atom, and `e` is answered there.
            // Complexity tradeoff: from `w` alone, the earlier method, one row `x - 1` paired
            // with k rows `y - w{j}` read all k pairs of each to free the atom `x * w{j}`,
            // Θ(k²) for the k pairs (34.1 times the instructions at 8 times the atoms, the
            // benchmark `products_whose_premise_holds_k_atoms_in_no_product_scale`); from the
            // open `z` alone, a long row with many open variables beside a short row holding
            // `w` costs the same the other way. With only those two, the weights, which count
            // every pair on a row, dead or alive, could make both dear at once (XR-presburger4
            // round 1, Claude F3, A4): with a sum `w0 + ... + w{k-1}` paired with a row `c{j}`
            // for each product `w{j} * c{j}` the formulas mention, beside `x - 1` paired with
            // each `y - w{j}` for `x * y`, each `w{j}` is on the k pairs of the sum and `x` on
            // k pairs, so the test of `{x - 1, y - w{j}}` read k pairs either way, Θ(k²) (61.8
            // times at 8k on the review's port; counting only live pairs gave 61.6, as the
            // pairs of the sum stay live). The third way reads there the one pair of `x - 1`
            // with each of the two rows on `w{j}`, O(log P). The cheapest of the three is taken,
            // O(min(...) × log P) per `w` for P pairs. Test
            // `tests::a_sum_beside_its_products_reads_few_pairs` (fails for either of the first
            // two ways alone); benchmark `products_beside_a_sum_whose_atoms_have_products_scale`
            // in `l01_38_presburger::bench::ladder`.
            let open = || {
                (self.vars[p].iter())
                    .map(|&(z, _)| z)
                    .filter(|z| !not.contains(z))
            };
            let (mut by_z, mut by_rows) = (0usize, 0usize);
            let held_w = self.holding[w].len();
            for z in open() {
                self.charge(1);
                by_z = by_z.saturating_add(self.weight[z]);
                by_rows = by_rows.saturating_add(held_w.saturating_mul(self.holding[z].len()));
                if by_z > self.weight[w] && by_rows > self.weight[w] {
                    break;
                }
            }
            if by_rows <= self.weight[w] && by_rows <= by_z {
                for z in open() {
                    let blocker = (self.holding[w].iter()).find_map(|&r| {
                        (self.holding[z].iter()).find_map(|&s| {
                            self.pairs_of(r, s)
                                .find(|&f| f != e && live[f] && self.holds(r, s, w, z))
                        })
                    });
                    let Some(f) = blocker else {
                        return true;
                    };
                    watch_on(watch, f, e);
                }
                continue 'w;
            }
            if by_z <= self.weight[w] {
                for z in open() {
                    let blocker = (self.holding[z].iter()).find_map(|&s| {
                        if !self.read() {
                            return None;
                        }
                        (self.opp[s].iter())
                            .take_while(|_| self.read())
                            .find(|&&(r, f)| f != e && live[f] && self.holds(r, s, w, z))
                            .map(|&(_, f)| f)
                    });
                    let Some(f) = blocker else {
                        return true;
                    };
                    watch_on(watch, f, e);
                }
                continue 'w;
            }
            // First, a live pair `{r, p}` other than `e` with `r` holding `w` that holds every
            // `w * z` of `p` at once: when `p` does not hold `w`, or no `z != w` shared by `r`
            // and `p` cancels (`r_w p_z + r_z p_w != 0`). Proof: the coefficient of `w * z` in
            // `r ⊗ p` is `r_w p_z + r_z p_w` for `z != w` and `r_w p_w` for `z = w`. For `z != w`
            // on `p`: if `p_w = 0` or `z` is not on `r` (`r_z = 0`), it is `r_w p_z != 0`;
            // otherwise `z` is shared and the test reads it nonzero. For `z = w` (then `p`
            // holds `w`) it is `r_w p_w != 0`. QED. That pair blocks `w` alone, and is watched
            // once. The pair of each `r` with `p` is found by binary search
            // ([`Index::pairs_of`]), O(log P) per row holding `w`, and the shared variables are
            // read through the shorter row ([`Index::cancels_nothing`]). Before XR-presburger4
            // META4 order 2 (B-L2) the test asked that `r` and `p` share no variable but `w`,
            // which missed the sum `x + z + y + w0 + ...` of Gemini's S1_plus_y family, sharing
            // `x` and `y` with each `x + (j + 1) y` without cancelling: Θ(k²) reads again
            // (62.80 times at 8k, computed by Gemini and META4). Test
            // `tests::a_sum_holding_the_goal_and_y_reads_few_pairs`. Complexity tradeoff: found
            // instead in the scan below, among the pairs of each `r`, with an entry in `watch`
            // for each `z` put into `not`, a sum `x + z + w0 + ... + w{k-1}` paired with itself
            // and with k rows `x + (j + 1) y` under the goal `x * z` read the k pairs of the sum's
            // row for each of the k pairs and entered each pair k times in `watch`: Θ(k²), 60.3
            // times the reads and 62.9 times the entries at 8k (XR-presburger4 round 3, Gemini:
            // 60.8 times). Test `tests::a_sum_holding_the_goal_reads_few_pairs`.
            let p_holds_w = self.coef(p, w).is_some();
            for &r in &self.holding[w] {
                let other_live = |f: usize| f != e && live[f];
                if let Some(f) = self.pairs_of(r, p).find(|&f| other_live(f))
                    && (!p_holds_w || self.cancels_nothing(r, p, w))
                {
                    watch_on(watch, f, e);
                    continue 'w;
                }
            }
            // No pair `{r, p}` blocks every `z` (above), so each pair read blocks only the `z`
            // that its rows share with `p`, and is watched once whatever their number.
            for &r in &self.holding[w] {
                if !self.read() {
                    return true;
                }
                for &(s, f) in &self.opp[r] {
                    if !self.read() {
                        return true;
                    }
                    if f == e || !live[f] {
                        continue;
                    }
                    let mut entered = false;
                    for z in self.shared(s, p) {
                        if !not.contains(&z) && self.holds(r, s, w, z) {
                            not.insert(z);
                            if !entered {
                                watch_on(watch, f, e);
                                entered = true;
                            }
                            if not.len() == width {
                                continue 'w;
                            }
                        }
                    }
                }
            }
            if not.len() < width {
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::collections::HashMap;

    use super::super::Refusal;
    use super::super::formula::{Atom, Factor, Lin};
    use super::{INPUT, PER_INPUT, SPENT, kept};

    /// The pairs kept, found by expanding every product: each round drops every live pair
    /// that alone holds an atom with no entry in `partners`, until a round drops none.
    fn expanded(
        rows: &[&Lin],
        pairs: &[(usize, usize)],
        partners: &HashMap<Factor, Vec<Factor>>,
    ) -> Vec<bool> {
        let free = |a: &Atom| match a {
            Atom::Product(u, v) => !partners.get(u).is_some_and(|ps| ps.contains(v)),
            _ => false,
        };
        let products: Vec<Lin> = (pairs.iter())
            .map(|&(a, b)| rows[a].product(rows[b]).expect("linear rows"))
            .collect();
        let atoms: Vec<Vec<&Atom>> = (products.iter())
            .map(|p| {
                (p.terms.iter())
                    .filter(|&(a, c)| !c.is_zero() && free(a))
                    .map(|(a, _)| a)
                    .collect()
            })
            .collect();
        let mut live = vec![true; pairs.len()];
        loop {
            let mut holders: HashMap<&Atom, usize> = HashMap::new();
            for (e, held) in atoms.iter().enumerate() {
                if live[e] {
                    for &a in held {
                        *holders.entry(a).or_default() += 1;
                    }
                }
            }
            let drop: Vec<usize> = (0..pairs.len())
                .filter(|&e| live[e] && atoms[e].iter().any(|&a| holders[a] == 1))
                .collect();
            if drop.is_empty() {
                return live;
            }
            for e in drop {
                live[e] = false;
            }
        }
    }

    /// The row `constant + Σ c × name`.
    fn row(constant: i64, terms: &[(i64, &str)]) -> Lin {
        let mut l = Lin::constant(constant);
        for &(c, name) in terms {
            l.plus_assign(&Lin::atom(Atom::Sym(name.into())).times(c));
        }
        l
    }

    /// The partners of the products `u * v` of `products`, both ways.
    fn partners_of(products: &[(&str, &str)]) -> HashMap<Factor, Vec<Factor>> {
        let mut partners: HashMap<Factor, Vec<Factor>> = HashMap::new();
        for &(u, v) in products {
            let (u, v) = (Factor::Sym(u.into()), Factor::Sym(v.into()));
            partners.entry(u.clone()).or_default().push(v.clone());
            partners.entry(v).or_default().push(u);
        }
        for ps in partners.values_mut() {
            zed_collections::sorted::sort_dedup(ps);
        }
        partners
    }

    #[test]
    fn an_atom_held_by_two_products_of_different_pairs_keeps_both() {
        // `c * d` is in `(1 - a - d) * (1 + a - b + c)` and in `(-a - b - d) * (-c)`, the
        // products for `a * b` and for `b * c`; every other atom of the two is mentioned or
        // cancels against a third product, so all four are kept, as the proof of
        // `l01_38_presburger::ladder::a_product_atom_cancels_through_the_products_of_another_pair`
        // needs.
        let rows = [
            row(1, &[(-1, "a"), (-1, "d")]),
            row(1, &[(1, "a"), (-1, "b"), (1, "c")]),
            row(0, &[(1, "b"), (-1, "a")]),
            row(0, &[(-1, "a"), (-1, "b"), (-1, "d")]),
            row(0, &[(-1, "c")]),
        ];
        let rows: Vec<&Lin> = rows.iter().collect();
        let pairs = [(0, 1), (2, 3), (3, 4), (4, 4)];
        let partners = partners_of(&[("a", "b"), ("b", "b"), ("b", "c"), ("c", "c")]);
        assert_eq!(
            kept(&rows, &pairs, &partners).expect("within the meter"),
            [true; 4]
        );
        assert_eq!(expanded(&rows, &pairs, &partners), [true; 4]);
        // Without the product of `-a - b - d` and `-c`, `(1 - a - d) * (1 + a - b + c)` alone
        // holds `c * d`, and then `(b - a) * (-a - b - d)` alone holds `a * d`.
        assert_eq!(
            kept(&rows, &[(0, 1), (2, 3), (4, 4)], &partners).expect("within the meter"),
            [false, false, true]
        );
    }

    #[test]
    fn an_atom_whose_coefficient_is_zero_frees_nothing() {
        // `(x + y) * (x - y)` is `x * x - y * y`: `x * y` cancels inside it, so with both
        // squares mentioned it is kept; with `y * y` not mentioned it goes.
        let rows = [
            row(0, &[(1, "x"), (1, "y")]),
            row(0, &[(1, "x"), (-1, "y")]),
        ];
        let rows: Vec<&Lin> = rows.iter().collect();
        let both = partners_of(&[("x", "x"), ("y", "y")]);
        assert_eq!(
            kept(&rows, &[(0, 1)], &both).expect("within the meter"),
            [true]
        );
        let one = partners_of(&[("x", "x")]);
        assert_eq!(
            kept(&rows, &[(0, 1)], &one).expect("within the meter"),
            [false]
        );
    }

    #[test]
    fn the_pairs_kept_are_those_left_by_expanding_every_product() {
        // Random rows over a few symbols, random pairs and random mentioned products, the
        // pairs given in a random order: `kept` agrees with `expanded`, so its answer does not
        // depend on the order of the pairs.
        let mut rng = crate::seed::Rng::new(0x2545_f491_4f6c_dd1d);
        let mut next = move |n: usize| rng.below(n);
        let names = ["a", "b", "c", "d", "e", "f"];
        let mut dropped = 0;
        for case in 0..4000 {
            let nv = 3 + next(4);
            let rows: Vec<Lin> = (0..2 + next(5))
                .map(|_| {
                    let terms: Vec<(i64, &str)> = (0..1 + next(4))
                        .map(|_| ([-2, -1, 1, 1, 2][next(5)], names[next(nv)]))
                        .collect();
                    row(next(5) as i64 - 2, &terms)
                })
                .filter(|r| !r.terms.is_empty())
                .collect();
            let n = rows.len();
            if n == 0 {
                continue;
            }
            let mut pairs: Vec<(usize, usize)> = Vec::new();
            for a in 0..n {
                for b in a..n {
                    if next(3) != 0 {
                        pairs.push((a, b));
                    }
                }
            }
            for i in (1..pairs.len()).rev() {
                pairs.swap(i, next(i + 1));
            }
            let products: Vec<(&str, &str)> = (0..next(5))
                .map(|_| (names[next(nv)], names[next(nv)]))
                .collect();
            let partners = partners_of(&products);
            let rows: Vec<&Lin> = rows.iter().collect();
            let want = expanded(&rows, &pairs, &partners);
            assert_eq!(
                kept(&rows, &pairs, &partners).expect("within the meter"),
                want,
                "case {case}: rows {rows:?}, pairs {pairs:?}, products {products:?}"
            );
            dropped += want.iter().filter(|&&k| !k).count();
        }
        assert!(dropped > 0, "no case dropped a pair");
    }

    /// The rows, pairs and products of a sum `w0 + ... + w{k-1} >= 0` paired with `c{j} >= 0`
    /// for each product `w{j} * c{j}`, and `y - w{j} >= 0` paired with `c{j}` and with
    /// `x - 1 >= 0` for `x * y`.
    #[allow(clippy::type_complexity)]
    fn sum_beside_products(k: usize) -> (Vec<Lin>, Vec<(usize, usize)>, Vec<(String, String)>) {
        let w = |j: usize| format!("w{j}");
        let c = |j: usize| format!("c{j}");
        let sum: Vec<(i64, String)> = (0..k).map(|j| (1, w(j))).collect();
        let sum: Vec<(i64, &str)> = sum.iter().map(|(c, s)| (*c, s.as_str())).collect();
        let mut rows = vec![row(-1, &[(1, "x")]), row(0, &sum)];
        let mut pairs = Vec::new();
        let mut products = vec![("x".to_string(), "y".to_string())];
        for j in 0..k {
            let a = rows.len();
            rows.push(row(0, &[(1, "y"), (-1, w(j).as_str())]));
            rows.push(row(0, &[(1, c(j).as_str())]));
            pairs.extend([(0, a), (1, a + 1), (a, a + 1)]);
            products.push((w(j), c(j)));
        }
        (rows, pairs, products)
    }

    /// The pairs read in deciding `sum_beside_products(k)`, and whether the answer is right
    /// (against `expanded` when `check`).
    fn reads(k: usize, check: bool) -> u64 {
        let (rows, pairs, products) = sum_beside_products(k);
        let products: Vec<(&str, &str)> = (products.iter())
            .map(|(u, v)| (u.as_str(), v.as_str()))
            .collect();
        let partners = partners_of(&products);
        let rows: Vec<&Lin> = rows.iter().collect();
        super::READS.with(|n| n.set(0));
        let got = kept(&rows, &pairs, &partners).expect("within the meter");
        let n = super::READS.with(std::cell::Cell::get);
        if check {
            assert_eq!(got, expanded(&rows, &pairs, &partners), "k = {k}");
        }
        n
    }

    /// XR-presburger4 round 1 A4 (Claude F3, F4): on `sum_beside_products`, the test of
    /// `{x - 1, y - w{j}}` read the k pairs on `x` from `x`, or the k pairs of the sum from
    /// `w{j}`, Θ(k²) in all either way; the lookup of the pairs of `x - 1` with the two rows on
    /// `w{j}` reads one. Counted: the pairs read at k and 8k grow by at most 9 times, and stay
    /// under 20 a link (taking only the first or only the second way reads about k a link).
    #[test]
    fn a_sum_beside_its_products_reads_few_pairs() {
        for k in [1, 2, 5, 9] {
            reads(k, true);
        }
        let (small, large) = (reads(250, false), reads(2000, false));
        assert!(small <= 20 * 250, "{small} pairs read at k = 250");
        assert!(large <= 20 * 2000, "{large} pairs read at k = 2000");
        assert!(
            9 * small >= large,
            "{small} and {large} pairs read at k = 250 and 2000"
        );
    }

    /// The reads ([`super::READS`]) of [`super::super::products`], the producer of the rows
    /// and pairs of [`kept`], on `x - 1 >= 0`, `y >= 0`, the k rows `z + u{j} >= 0`, on no
    /// pair, and after them the k rows `y + (j + 1) z >= 0`, each paired with `x - 1` for the
    /// goal's `x * y`; with the products made. The rows on no pair come first, so a search of
    /// the rows holding `z` that walks them meets all k before the first row on a pair
    /// (`Formula::and` keeps the order of its parts); interleaved, it met one.
    fn reads_through_products(k: usize) -> (u64, Vec<super::super::formula::Formula>) {
        use super::super::formula::{Con, Formula};
        use super::super::{Relation, Tier, products};
        let ge0 = |l: Lin| Formula::Con(Con::ge(&l, &Lin::constant(0)));
        let mut premise = vec![ge0(row(-1, &[(1, "x")])), ge0(row(0, &[(1, "y")]))];
        for j in 0..k {
            let u = format!("u{j}");
            premise.push(ge0(row(0, &[(1, "z"), (1, u.as_str())])));
        }
        for j in 0..k {
            premise.push(ge0(row(0, &[(1, "y"), (j as i64 + 1, "z")])));
        }
        let xy = Lin::atom(Atom::product(&"x".into(), &"y".into()));
        let (p, g) = (Relation::of(Formula::and(premise)), Relation::of(ge0(xy)));
        super::READS.with(|n| n.set(0));
        let (made, _) = products(&p, &g, Tier::Cheap).expect("within the meter");
        (super::READS.with(std::cell::Cell::get), made)
    }

    /// XR-presburger4 round 2, L1 (Codex): through `products`, the rows on no pair among the
    /// rows holding `z` are not walked, so the reads (pairs, lookups and lists entered, empty
    /// lists too) at k and 8k grow by at most 9 times and stay under 20 a row. With every row
    /// kept in `holding`, the test of each pair `{x - 1, y + (j + 1) z}` entered the k empty
    /// lists of the rows `z + u{j}`: Θ(k²). That dropping those rows from `holding` changes no
    /// pair kept is checked against `expanded` by
    /// `the_pairs_kept_are_those_left_by_expanding_every_product`, whose random pairs leave
    /// some rows on no pair. Red with the change removed: 63,002 and 4,004,002 reads (VM hm1,
    /// job 38). On the round-2 commit 2e4a53cd0 it passes (251 and 2,001), not because round
    /// 2 skipped those lists: it entered the k empty lists in each of the k tests too, but its
    /// counter counted only pairs read, not lists entered (XR-presburger4 round 3, Claude and
    /// Codex: counted, 63,002 and 4,004,002 there as well).
    #[test]
    fn a_premise_of_rows_on_no_pair_reads_few_lists() {
        let ((small, made), large) = (reads_through_products(250), reads_through_products(2000).0);
        assert!(!made.is_empty(), "no product made at k = 250");
        assert!(small > 0, "no read counted at k = 250");
        assert!(small <= 20 * 250, "{small} reads at k = 250");
        assert!(large <= 20 * 2000, "{large} reads at k = 2000");
        assert!(
            9 * small >= large,
            "{small} and {large} reads at k = 250 and 2000"
        );
    }

    /// Gemini's grid family (XR-presburger4 round 2, L6) through [`super::super::products`]:
    /// for each `i < m` the rows `x - i >= 0`, `y - w{i} >= 0`, `c{i} >= 0` and
    /// `w0 + ... + w{m-1} - i >= 0`, and the goal `x * y + sum_i w{i} * c{i} >= 0`. As written
    /// the rows `x - i` are parallel, and so are the sums: `Formula::and` keeps the tightest
    /// of each, so `x * w{j}` and `w{l} * c{j}` each have one holder and no pair is kept. With
    /// `fresh`, the rows are `x + i s >= 0` and `w0 + ... + w{m-1} + i t >= 0`, `s` and `t`
    /// fresh, no two parallel (the reviewer's variant): the pairs are the bicliques of the rows
    /// on `x` and `y` and of those on each `w{i}` and `c{i}`, Θ(m²) = Θ(k) for the input size
    /// `k`, and the kept products hold Θ(m³) = Θ(k^{3/2}) terms. The reads, the products made
    /// and their terms, or the refusal; the steps charged and the input of [`kept`] ([`SPENT`],
    /// [`INPUT`]), zero when it did not run.
    fn grid_through_products(
        m: usize,
        fresh: bool,
    ) -> (u64, Result<(usize, usize), Refusal>, u64, u64) {
        use super::super::formula::{Con, Formula};
        use super::super::{Relation, Tier, products};
        let ge0 = |l: Lin| Formula::Con(Con::ge(&l, &Lin::constant(0)));
        let mut premise = Vec::new();
        let mut goal = Lin::atom(Atom::product(&"x".into(), &"y".into()));
        let w: Vec<String> = (0..m).map(|i| format!("w{i}")).collect();
        let sum: Vec<(i64, &str)> = w.iter().map(|v| (1, v.as_str())).collect();
        for (i, wi) in w.iter().enumerate() {
            let ci = format!("c{i}");
            let k = i as i64;
            let (xi, gi) = match (fresh, i) {
                (true, 0) | (false, _) => (row(-k, &[(1, "x")]), row(-k, &sum)),
                (true, _) => (
                    row(0, &[(1, "x"), (k, "s")]),
                    row(0, &sum).plus(&row(0, &[(k, "t")])),
                ),
            };
            premise.push(ge0(xi));
            premise.push(ge0(row(0, &[(1, "y"), (-1, wi.as_str())])));
            premise.push(ge0(row(0, &[(1, ci.as_str())])));
            premise.push(ge0(gi));
            goal = goal.plus(&Lin::atom(Atom::product(
                &wi.as_str().into(),
                &ci.as_str().into(),
            )));
        }
        let (p, g) = (Relation::of(Formula::and(premise)), Relation::of(ge0(goal)));
        super::READS.with(|n| n.set(0));
        SPENT.with(|n| n.set(0));
        INPUT.with(|n| n.set(0));
        let made = products(&p, &g, Tier::Cheap).map(|(made, _)| made);
        fn terms(f: &Formula) -> usize {
            match f {
                Formula::Con(c) => c.lin.terms.len(),
                Formula::Not(g) => terms(g),
                Formula::And(fs) | Formula::Or(fs) => fs.iter().map(terms).sum(),
            }
        }
        let made = made.map(|made| (made.len(), made.iter().map(terms).sum()));
        let reads = super::READS.with(std::cell::Cell::get);
        (reads, made, SPENT.with(Cell::get), INPUT.with(Cell::get))
    }

    /// XR-presburger4 round 2, L6 (Gemini): the grid family ([`grid_through_products`]). As
    /// Gemini wrote it, `products` cannot build it: its parallel rows are one row each by then,
    /// and no product is made. Its variant with no two rows parallel is built (at least m²
    /// pairs), and there the reads of [`kept`] grew past the scaling limit when it ran to the
    /// end: from m = 16 to 32 the products made grew 4 times (512 to 2,048), the terms they
    /// hold 7.1 times (5,328 to 37,792), and the reads 13.2 times (14,473 to 191,505). With
    /// the meter ([`PER_INPUT`]) the question is refused ([`Refusal::Products`]) once the
    /// steps pass the limit, so the steps are at most [`PER_INPUT`] times the input of
    /// [`kept`] plus the steps of one read, and grow with the input: at m and about 8 times
    /// its input (m = 45 and 132) by at most 9 times. The counts are deterministic and held
    /// exactly (XR-presburger4 round 3, Claude: a ratio alone let a method reading
    /// Θ(products²) pass). Red without the meter: the question is answered, not refused.
    /// Whether a method decides this family in O(input log input) is plan row
    /// PRES-pairs-grid-reads (open).
    #[test]
    fn the_meter_refuses_the_grid_family_in_steps_linear_in_its_input() {
        let mut counts = Vec::new();
        for m in [45, 132] {
            let (_, made, _, _) = grid_through_products(m, false);
            assert_eq!(
                made,
                Ok((0, 0)),
                "m = {m}: products made from parallel rows"
            );
            let (reads, made, spent, input) = grid_through_products(m, true);
            assert_eq!(made, Err(Refusal::Products), "m = {m}");
            assert!(
                spent > PER_INPUT * input,
                "m = {m}: {spent} steps, input {input}"
            );
            assert!(
                spent <= PER_INPUT * input + input,
                "m = {m}: {spent} steps, input {input}"
            );
            counts.push((reads, spent, input));
        }
        // Measured (job 47): the steps 8.11 times at 8.11 times the input, the reads 8.18 times.
        assert_eq!(
            counts,
            [(104_841, 266_422, 6_660), (857_383, 2_159_522, 53_988)]
        );
        assert!(9 * counts[0].1 >= counts[1].1, "{counts:?}");
        assert!(8 * counts[0].2 <= counts[1].2, "{counts:?}");
    }

    /// The meter's refusal reaches the caller of [`super::super::entails`] from each tier of
    /// [`super::super::products`]: `x >= 7`, which the goal `x * y >= 7 y` needs with `y >= 0`,
    /// follows from `x + b >= 10`, `x - b - c >= 4` and `c >= 0` together, where `b` and `c`
    /// are factors (of `b * c >= -1000000`), so it is no cheap bound and the goal is proved at
    /// `Tier::Local`. A refusal at `Tier::Cheap` was returned before; one at `Tier::Local` or
    /// `Tier::Whole` was taken for `NotProved` (XR-presburger4 META3, coordinator point 3),
    /// which says the linear reading fails with every product allowed, never checked. Red
    /// with that: `NotProved` at the second call.
    #[test]
    fn entailment_refused_for_its_products_says_so() {
        use super::super::formula::{Con, Formula};
        use super::super::{Entailment, Relation, entails};
        let ge0 = |l: Lin| Formula::Con(Con::ge(&l, &Lin::constant(0)));
        let bc = Lin::atom(Atom::product(&"b".into(), &"c".into()));
        let premise = vec![
            ge0(row(0, &[(1, "z")])),
            ge0(row(2, &[(-1, "z")])),
            ge0(row(0, &[(1, "x"), (-1, "z")])),
            ge0(row(0, &[(1, "y"), (-1, "z")])),
            ge0(row(-10, &[(1, "x"), (1, "b")])),
            ge0(row(-4, &[(1, "x"), (-1, "b"), (-1, "c")])),
            ge0(row(0, &[(1, "c")])),
            ge0(bc.plus(&Lin::constant(1_000_000))),
        ];
        let xy = Lin::atom(Atom::product(&"x".into(), &"y".into()));
        let goal = ge0(xy.minus(&row(0, &[(7, "y")])));
        let (p, g) = (Relation::of(Formula::and(premise)), Relation::of(goal));
        assert!(matches!(entails(&p, &g), Entailment::Proved(_)));
        let refused_after = |p: &Relation, g: &Relation, calls: usize| {
            super::REFUSE_AFTER.with(|c| c.set(Some(calls)));
            let got = entails(p, g);
            super::REFUSE_AFTER.with(|c| c.set(None));
            got
        };
        for (calls, tier) in [(0, "Cheap"), (1, "Local")] {
            let got = refused_after(&p, &g, calls);
            assert_eq!(got, Entailment::Refused(Refusal::Products), "{tier}");
        }
        // `Tier::Whole` (XR-presburger4 META4 order 4, NB-L1): `x >= 7` follows from
        // `x + 2 z >= 21` and the difference rows `z <= w <= x` together, as `3 x >= 21`. The
        // cluster of `x` holds the general row alone, as `w` is outside it, so neither its rows
        // nor the bounds give `x` a lower side, and only the hull over the whole part does. With
        // the third call of `kept` refused the answer is the refusal; with the fourth, none is
        // made, so the third is the last, at `Tier::Whole`, and the goal is proved there.
        let premise = vec![
            ge0(row(-21, &[(1, "x"), (2, "z")])),
            ge0(row(0, &[(1, "w"), (-1, "z")])),
            ge0(row(0, &[(1, "x"), (-1, "w")])),
            ge0(row(0, &[(1, "y")])),
        ];
        let p = Relation::of(Formula::and(premise));
        assert!(matches!(entails(&p, &g), Entailment::Proved(_)));
        assert!(matches!(refused_after(&p, &g, 3), Entailment::Proved(_)));
        let got = refused_after(&p, &g, 2);
        assert_eq!(got, Entailment::Refused(Refusal::Products), "Whole");
    }

    /// Gemini's family (XR-presburger4 round 3) through [`super::super::products`]: the row
    /// `x + z + w0 + ... + w{k-1} >= 0`, the k rows `x + (j + 1) y >= 0` and the goal
    /// `x * z >= 0`, so the pairs are the sum with itself and with each row `x + (j + 1) y`.
    /// The reads, the entries made in the lists of blockers ([`super::WATCHED`]) and the
    /// number of products made, or the refusal.
    fn sum_holding_the_goal(k: usize) -> (u64, u64, Result<usize, Refusal>) {
        sum_holding_the_goal_and(k, false)
    }

    /// [`sum_holding_the_goal`], with `y` in the sum too when `with_y` (Gemini's S1_plus_y,
    /// XR-presburger4 round 4, B-L2).
    fn sum_holding_the_goal_and(k: usize, with_y: bool) -> (u64, u64, Result<usize, Refusal>) {
        use super::super::formula::{Con, Formula};
        use super::super::{Relation, Tier, products};
        let ge0 = |l: Lin| Formula::Con(Con::ge(&l, &Lin::constant(0)));
        let w: Vec<String> = (0..k).map(|j| format!("w{j}")).collect();
        let mut sum: Vec<(i64, &str)> = vec![(1, "x"), (1, "z")];
        if with_y {
            sum.push((1, "y"));
        }
        sum.extend(w.iter().map(|v| (1, v.as_str())));
        let mut premise = vec![ge0(row(0, &sum))];
        for j in 0..k {
            premise.push(ge0(row(0, &[(1, "x"), (j as i64 + 1, "y")])));
        }
        let xz = Lin::atom(Atom::product(&"x".into(), &"z".into()));
        let (p, g) = (Relation::of(Formula::and(premise)), Relation::of(ge0(xz)));
        super::READS.with(|n| n.set(0));
        super::WATCHED.with(|n| n.set(0));
        let made = products(&p, &g, Tier::Cheap).map(|(made, _)| made.len());
        let reads = super::READS.with(std::cell::Cell::get);
        (reads, super::WATCHED.with(std::cell::Cell::get), made)
    }

    /// XR-presburger4 round 3 (Gemini): on [`sum_holding_the_goal`] the reads, the entries in
    /// the lists of blockers and the products made grow by at most 9 times at 8 times `k`, and
    /// are exact (deterministic). Before the lookup of a pair `{r, p}` blocking every `w * z`
    /// in [`Index::frees`], the test of each pair of the sum read the k pairs of the sum's row
    /// and entered its blocker once for each `z`: 10,706 and 645,606 reads (60.3 times), 10,201
    /// and 641,601 entries (62.9 times), red here (VM v5e2, job 42); with the entries made once
    /// a blocker but no lookup, the entries were 301 and 2,401 and the reads unchanged, also
    /// red. The k products kept, of the sum with each row `x + (j + 1) y`, hold Θ(k²) terms
    /// (3 (k + 3) each), past [`PER_INPUT`] times the input, so since XR-presburger4 META3
    /// (L6) [`super::super::products`] refuses them before they are expanded, after
    /// [`kept`] has chosen them in the reads counted here.
    #[test]
    fn a_sum_holding_the_goal_reads_few_pairs() {
        let (small, large) = (sum_holding_the_goal(100), sum_holding_the_goal(800));
        let refused = Err(Refusal::Products);
        assert_eq!(
            (small.clone(), large.clone()),
            ((708, 201, refused.clone()), (5_608, 1_601, refused))
        );
        assert!(large.0 <= 9 * small.0, "reads {} and {}", small.0, large.0);
        assert!(
            large.1 <= 9 * small.1,
            "entries {} and {}",
            small.1,
            large.1
        );
    }

    /// B-L2 (XR-presburger4 META4 order 2): with `y` in the sum, the sum shares `x` and `y`
    /// with each row `x + (j + 1) y`, and no atom cancels (`1 · (j + 1) + 1 · 1 != 0`), so
    /// each pair of the sum with a row still blocks every atom of the sum at once. The reads
    /// and the entries in the lists of blockers grow by at most 9 times at 8 times `k`. Red
    /// before [`Index::cancels_nothing`] replaced the test that `r` and `p` share no variable
    /// but `w`: the pair test fell through to the scan of every pair of each row on `w`.
    #[test]
    fn a_sum_holding_the_goal_and_y_reads_few_pairs() {
        let within = |k: usize| {
            let got = sum_holding_the_goal_and(k, true);
            let (spent, input) = (SPENT.with(Cell::get), INPUT.with(Cell::get));
            // The meter stops a pass early, which would hide its reads: it must not have.
            assert!(spent <= PER_INPUT * input, "k = {k}: {spent} steps on input {input}");
            got
        };
        let (small, large) = (within(100), within(800));
        assert!(large.0 <= 9 * small.0, "reads {} and {}", small.0, large.0);
        assert!(large.1 <= 9 * small.1, "entries {} and {}", small.1, large.1);
    }

    /// Codex's family (XR-presburger4 round 4, L4-B, `ladder_empty_r4.rs`): the k rows
    /// `x + (i + 1) s >= 0` and the goal `x * y0 + ... + x * y{k-1} >= 0`, no `y{j}` on any
    /// row. The rows [`super::super::products`] walks ([`super::WALKED`]) and the products
    /// made.
    fn products_of_a_factor_on_no_row(k: usize) -> (u64, Result<usize, Refusal>) {
        use super::super::formula::{Con, Formula};
        use super::super::{Relation, Tier, products};
        let ge0 = |l: Lin| Formula::Con(Con::ge(&l, &Lin::constant(0)));
        let premise: Vec<Formula> = (0..k)
            .map(|i| ge0(row(0, &[(1, "x"), (i as i64 + 1, "s")])))
            .collect();
        let goal = (0..k).fold(Lin::constant(0), |sum, j| {
            sum.plus(&Lin::atom(Atom::product(
                &"x".into(),
                &format!("y{j}").as_str().into(),
            )))
        });
        let (p, g) = (Relation::of(Formula::and(premise)), Relation::of(ge0(goal)));
        super::WALKED.with(|n| n.set(0));
        let made = products(&p, &g, Tier::Cheap).map(|(made, _)| made.len());
        (super::WALKED.with(std::cell::Cell::get), made)
    }

    /// L4-B: a product `x * y{j}` with no row on `y{j}` forms no pair, so no row on `x` is
    /// walked for it. Red before the check of an empty side in [`super::super::products`]:
    /// every row on `x` was walked for each such product, k² rows, 62,500 and 4,000,000 at
    /// k = 250 and 2,000 (Codex, computed; reproduced here).
    #[test]
    fn products_of_a_factor_on_no_row_walk_no_rows() {
        for k in [250, 2_000] {
            assert_eq!(products_of_a_factor_on_no_row(k), (0, Ok(0)), "k = {k}");
        }
    }
}
