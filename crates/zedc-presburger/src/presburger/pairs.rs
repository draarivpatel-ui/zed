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
//! The coefficient of `w * z` in `p ⊗ q` is `p_w q_z + p_z q_w`, and `p_w q_w` for `w * w`.
//! The method is Neon's three stages (XR-presburger4, archive/neon/NEON.md and its model
//! pairs_model.rs), with the repairs of round 4 (META4 order 3): stage 1 finds pairs that can
//! never go ([`certify`]), stage 2 drops pairs by local counts without expanding any product,
//! and stage 3 expands what is left and peels it exactly ([`settle`]). Every pass is bounded
//! before it runs by a sum computed from the input; none is stopped part way.

#[cfg(test)]
use std::cell::Cell;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

use zedc_int::int::Int;

use super::Refusal;
use super::formula::{Factor, Lin};

/// The work allowed to each optional pass of stage 1 ([`certify`]: C2's intersections, C4's
/// subset tests and C5's neighbor cores) for each unit of the input `k`. A pass whose cost,
/// computed before it runs, is above `CERTIFY_PER_INPUT × k` is skipped whole: skipping it only
/// certifies fewer pairs, which leaves more to stage 3, so it can change whether a question is
/// refused (the class Φ in [`kept`]) but never the pairs kept. The value is Neon's (NEON.md,
/// "4k"), the smallest that passes C5 on the grid family, whose neighbor sum is about 3k.
pub(super) const CERTIFY_PER_INPUT: usize = 4;

/// The terms stage 3 may expand for each unit of the input `k` ([`settle`]): the products of
/// the pairs left after stages 1 and 2, and the work of testing their atoms against the
/// certified pairs. Above it the question is refused (Φ in [`kept`]). The value is Neon's
/// (NEON.md, β = 16): the families of the crate's tests that reach stage 3 expand at most about
/// 4k terms there (the cascade chain, 4 terms for each of its k pairs).
pub(super) const EXPAND_PER_INPUT: usize = 16;

#[cfg(test)]
thread_local! {
    /// Tests only: the units of work done by [`kept`] on this thread (each pass of an inner
    /// loop counts one), summed over calls until reset.
    pub(super) static WORK: Cell<u64> = const { Cell::new(0) };
    /// Tests only: the input `k = R + T + P + M` of the last call of [`kept`] on this thread.
    pub(super) static INPUT: Cell<u64> = const { Cell::new(0) };
    /// Tests only: whether the last call of [`kept`] answered (`Some(true)`) or refused.
    pub(super) static ANSWERED: Cell<Option<bool>> = const { Cell::new(None) };
    /// Tests only: `Some(n)` makes the call of [`kept`] after the next `n` on this thread
    /// refuse at once; `None` leaves every call alone.
    pub(super) static REFUSE_AFTER: Cell<Option<usize>> = const { Cell::new(None) };
    /// Tests only: the tier [`super::products`] runs at, set before it calls [`kept`].
    pub(super) static TIER: Cell<Option<super::Tier>> = const { Cell::new(None) };
    /// Tests only: the tier of the call that [`REFUSE_AFTER`] refused.
    pub(super) static REFUSED_AT: Cell<Option<super::Tier>> = const { Cell::new(None) };
    /// Tests only: the rows on the first factor of a wanted product that
    /// [`super::products`] walked to form pairs, on this thread.
    pub(super) static WALKED: Cell<u64> = const { Cell::new(0) };
    /// Tests only: the passes turned off ([`off`]), for the negative controls of the tests.
    pub(super) static OFF: Cell<u32> = const { Cell::new(0) };
}

/// The passes a test can turn off (`OFF`, in tests); outside tests [`off`] is false for each.
pub(super) mod pass {
    pub(crate) const C1: u32 = 1;
    pub(crate) const C2: u32 = 2;
    pub(crate) const C3: u32 = 4;
    pub(crate) const C4: u32 = 8;
    pub(crate) const C5: u32 = 16;
    pub(crate) const P2: u32 = 32;
    /// The test of P3 that no representative has the pair's two rows.
    pub(crate) const P3_GUARD: u32 = 64;
    pub(crate) const P3: u32 = 128;
    pub(crate) const P4: u32 = 256;
}

/// Whether a test turned off the pass `bit` ([`OFF`]).
#[cfg(test)]
fn off(bit: u32) -> bool {
    OFF.with(|o| o.get() & bit != 0)
}

#[cfg(not(test))]
fn off(_: u32) -> bool {
    false
}

/// Counts `n` units of work ([`WORK`]), in tests; nothing otherwise.
fn tick(n: usize) {
    #[cfg(test)]
    WORK.with(|w| w.set(w.get().saturating_add(n as u64)));
    #[cfg(not(test))]
    let _ = n;
}

/// Counts one row walked by [`super::products`] ([`WALKED`]), in tests.
#[cfg(test)]
pub(super) fn walked() {
    WALKED.with(|n| n.set(n.get() + 1));
}

/// Whether each of `pairs`, the unordered pairs of indices into `rows`, is kept: the pairs left
/// when every pair whose product alone holds an atom with no entry in `partners` (the factors
/// each factor is multiplied by in a product the formulas mention) is dropped, to a fixpoint.
///
/// Method, for the input `k = R + T + P + M` (rows, their terms, pairs, entries of
/// `partners`):
/// 1. [`certify`] marks pairs that can never be dropped, and stands for all of them by a few
///    representative pairs whose atoms are exactly the atoms the marked pairs hold.
/// 2. [`settle`] drops pairs by local rules that need no product expanded (P0, P3, P2, P4 there).
/// 3. [`settle`] expands the pairs left (the core) and peels them exactly against the
///    representatives (Solver A of NEON.md).
///
/// Proof that the answer is the fixpoint, when there is one: stage 1 marks a set C of pairs
/// in which every free atom of a pair of C has at least two holders in C (each rule at its
/// code says why). No pair of C is ever dropped: the first to go would hold an atom alone,
/// but another pair of C, still live, holds it too. So the atoms of C are blocked for good,
/// and the representatives, which hold exactly those atoms, stand for C. Each drop of stages 2
/// and 3 is of a pair that holds a free atom no other live pair or representative holds (each
/// rule at its code), so it is a drop the rule allows; stage 3 ends when no core pair holds a
/// free atom alone. The set left is then one from which no pair can go, reached by allowed
/// drops, which by the module's proof is the one fixpoint.
///
/// Φ, the questions refused (`Err(Refusal::Products)`), exactly: with stage 1 run under its
/// three static conditions (C2, C4, C5 at [`certify`], each computed from the input before the
/// pass) and stage 2 to its end (P4 under its static condition, `W4 <= CERTIFY_PER_INPUT × k`
/// at [`settle`]), the core `E` left is not empty and either
/// `W_exp = Σ_{a, b ∈ E} |a| × |b| > EXPAND_PER_INPUT × k`, or some free atom of the core has
/// one holder in it and `W_exp + min(Q_rep, W_rep) > EXPAND_PER_INPUT × k`, where `W_rep` is
/// the sum of `|a| × |b|` over the representatives and `Q_rep` the sum over the core's free
/// atoms `w * z` of `min(rw(w), rw(z))`, `rw(v)` the number of representatives on the rows
/// that hold `v` (the test of each atom against the representatives reads that many). Proof:
/// these are the only two returns of `Err` ([`settle`], stage 3), and both are taken exactly
/// when their condition holds; nothing else refuses. Inside Φ are all inputs whose core holds
/// Θ(k²) terms: the triangle reduction below shows that no near-linear method is likely to
/// decide every input, so some such class must be refused; Φ is the one this method leaves.
/// Which families of the tests land outside Φ, and in which stage they are decided: the
/// `*_scale` tests below, each with a negative control that turns off the stage it needs.
///
/// Cost: O(k log k) in all. [`Rows::new`] O(T log T + M log M); stage 1 O((T + P) log P)
/// outside its three optional passes, each of which runs only when its cost, summed before,
/// is at most `CERTIFY_PER_INPUT × k` lookups of O(log k); stage 2 O((T + P) log P) (each
/// pair is removed once, in O(1) from the live lists by swap-remove, and each row's heap entry
/// is popped once; each variable's holders are walked once, when its count of live rows falls
/// to one), and P4 O(W4 log P) when it runs, `W4 <= CERTIFY_PER_INPUT × k`; stage 3 at most `EXPAND_PER_INPUT × k` terms and lookups, each O(log k), or the
/// refusal after O(|E|) to sum `W_exp`. Units of work are counted in tests ([`WORK`]).
///
/// Complexity tradeoff: the naive method expands every product, Σ |p| × |q| terms over the
/// pairs, Θ(k²) for one row of k factors paired with a row on each of them, such as one sum
/// `b0 + ... + b{k-1}` beside the products `b{i} * c{i}` (33.5 times the instructions at 8
/// times the links, C2-pairs-long-row), and then counts the holders of each atom again after
/// each round of drops. The method before this one tested each pair from its factor rows
/// under a meter of 40 steps a unit of input (PER_INPUT), and refused when the meter ran out:
/// whether a question was refused depended on a count of steps, and Gemini's grid family was
/// refused at every size (XR-presburger4 META4, L4-A: 45.02 steps a unit at every n). Here no
/// count of steps stops anything; Φ above is a condition on the input and the stages' result.
/// Triangle detection reduces to this function in linear size (XR-presburger4
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
/// primary sources). On the triangle reduction the rows `S{c}` are paired with rows of one
/// variable each, and C5 certifies the pairs `{u{a}, S{c}}` whose variables lie in the 2-core
/// of the neighbors of `S{c}`; what it leaves is refused when its expansion passes Φ.
/// Searched for a near-linear test of the pairs that alone hold an atom under deletions
/// (paperclip, 8 searches of 227 to 245 results each, 2026-11-22): none found. Gemini's grid
/// family, as written, cannot reach this function (its parallel rows are one row each after
/// `Formula::and`). Its variant with no two rows parallel does: under the meter it was refused
/// at every size; here C1 and C2 certify its bicliques, C4 the pairs whose rows lie inside a
/// certified one, and P2 drops the rest
/// (`tests::the_grid_family_is_decided_in_work_linear_in_its_input`).
/// Tests: `tests::the_pairs_kept_are_those_left_by_expanding_every_product` (against the naive
/// method), the `tests::*_scale` families, and
/// `l01_38_presburger::ladder::products_whose_factors_one_sum_on_the_factors_joins_scale`.
pub(super) fn kept(
    rows: &[&Lin],
    pairs: &[(usize, usize)],
    partners: &HashMap<Factor, Vec<Factor>>,
) -> Result<Vec<bool>, Refusal> {
    let k = rows.len()
        + rows.iter().map(|r| r.terms.len()).sum::<usize>()
        + pairs.len()
        + partners.values().map(Vec::len).sum::<usize>();
    #[cfg(test)]
    {
        INPUT.with(|c| c.set(k as u64));
        ANSWERED.with(|c| c.set(None));
        let refuse = REFUSE_AFTER.with(|c| match c.get() {
            Some(0) => {
                c.set(None);
                true
            }
            Some(n) => {
                c.set(Some(n - 1));
                false
            }
            None => false,
        });
        if refuse {
            REFUSED_AT.with(|c| c.set(TIER.with(Cell::get)));
            ANSWERED.with(|c| c.set(Some(false)));
            return Err(Refusal::Products);
        }
    }
    let rs = Rows::new(rows, partners);
    let (cert, reps) = certify(&rs, pairs, k);
    let rest: Vec<usize> = (0..pairs.len()).filter(|&e| !cert[e]).collect();
    let rest_pairs: Vec<(usize, usize)> = rest.iter().map(|&e| pairs[e]).collect();
    let left = settle(&rs, &rest_pairs, &reps, k);
    #[cfg(test)]
    ANSWERED.with(|c| c.set(Some(left.is_ok())));
    let left = left?;
    let mut out = cert;
    for (i, &e) in rest.iter().enumerate() {
        out[e] = left[i];
    }
    Ok(out)
}

/// The factor rows of [`kept`], with each factor numbered.
pub(super) struct Rows<'a> {
    /// Each row's variables, by number, ascending, with their coefficients (none zero).
    vars: Vec<Vec<(usize, &'a Int)>>,
    /// The variables each variable is multiplied by in a product the formulas mention,
    /// ascending, both ways.
    mentioned: Vec<Vec<usize>>,
    /// Whether all of a row's coefficients have one sign.
    mono: Vec<bool>,
}

impl<'a> Rows<'a> {
    pub(super) fn new(rows: &[&'a Lin], partners: &HashMap<Factor, Vec<Factor>>) -> Rows<'a> {
        let mut number: HashMap<Factor, usize> = HashMap::new();
        let vars: Vec<Vec<(usize, &Int)>> = (rows.iter())
            .map(|row| {
                let mut vs: Vec<(usize, &Int)> = (row.terms.iter())
                    .filter_map(|(a, c)| {
                        tick(1);
                        let f = Factor::of(a)?;
                        let next = number.len();
                        Some((*number.entry(f).or_insert(next), c))
                    })
                    .collect();
                vs.sort_unstable_by_key(|&(v, _)| v);
                vs
            })
            .collect();
        let mut mentioned: Vec<Vec<usize>> = vec![Vec::new(); number.len()];
        for (u, ps) in partners {
            let Some(&u) = number.get(u) else {
                continue;
            };
            for p in ps {
                tick(1);
                if let Some(&p) = number.get(p) {
                    mentioned[u].push(p);
                    mentioned[p].push(u);
                }
            }
        }
        for m in &mut mentioned {
            zed_collections::sorted::sort_dedup(m);
        }
        let mono = (vars.iter())
            .map(|vs| {
                let neg = vs.iter().filter(|(_, c)| c.is_negative()).count();
                neg == 0 || neg == vs.len()
            })
            .collect();
        Rows {
            vars,
            mentioned,
            mono,
        }
    }

    fn len(&self, r: usize) -> usize {
        self.vars[r].len()
    }

    /// The coefficient of `w` in row `r`, none when it has none.
    fn coef(&self, r: usize, w: usize) -> Option<&'a Int> {
        tick(1);
        zed_collections::sorted::sorted_get(&self.vars[r], &w).copied()
    }

    /// Whether the formulas mention `w * z`.
    fn is_mentioned(&self, w: usize, z: usize) -> bool {
        tick(1);
        self.mentioned[w].binary_search(&z).is_ok()
    }

    /// Whether the coefficient of `w * z` in `r ⊗ s` is not zero (symmetric in `r`, `s`).
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

    /// The number of variables `a` and `b` share, read through the shorter:
    /// O(min(|a|, |b|) log).
    fn shared(&self, a: usize, b: usize) -> usize {
        let (short, long) = if self.len(a) <= self.len(b) {
            (a, b)
        } else {
            (b, a)
        };
        (self.vars[short].iter())
            .filter(|&&(v, _)| self.coef(long, v).is_some())
            .count()
    }

    /// Whether `a ⊗ b` holds every atom `w * z`, `w` of `a` and `z` of `b`: no two shared
    /// variables cancel. Proof: for `z` not on `a` the coefficient is `a_w b_z != 0`, and for
    /// `w = z` it is `a_w b_w != 0`; for `w != z` both shared it is `a_w b_z + a_z b_w`, zero
    /// exactly when `a_w / b_w = -(a_z / b_z)`. So the product cancels exactly when two shared
    /// variables have ratios `a_v / b_v` of equal size and opposite sign; each ratio is kept
    /// reduced, as the pair `(|a_v| / g, |b_v| / g)`, `g` their gcd, with its sign.
    /// O(min(|a|, |b|) log) and one gcd per shared variable.
    fn no_cancel(&self, a: usize, b: usize) -> bool {
        let (short, long) = if self.len(a) <= self.len(b) {
            (a, b)
        } else {
            (b, a)
        };
        let mut seen: HashMap<(Int, Int), bool> = HashMap::new();
        for &(v, cs) in &self.vars[short] {
            let Some(cl) = self.coef(long, v) else {
                continue;
            };
            let (x, y) = (cs.abs(), cl.abs());
            let g = x.gcd(&y);
            let key = (x.div_floor(&g), y.div_floor(&g));
            let neg = cs.is_negative() != cl.is_negative();
            match seen.get(&key) {
                Some(&n) if n != neg => return false,
                Some(_) => {}
                None => {
                    seen.insert(key, neg);
                }
            }
        }
        true
    }

    /// The row `r` divided by the gcd of its coefficients, its first coefficient made
    /// positive: two rows have one direction exactly when one is a nonzero multiple of the
    /// other. O(|r|).
    fn direction(&self, r: usize) -> Vec<(usize, Int)> {
        let g = (self.vars[r].iter()).fold(Int::ZERO, |g, &(_, c)| g.gcd(c));
        let flip = self.vars[r].first().is_some_and(|(_, c)| c.is_negative());
        (self.vars[r].iter())
            .map(|&(v, c)| {
                tick(1);
                let q = c.div_floor(&g);
                (v, if flip { -&q } else { q })
            })
            .collect()
    }
}

/// A representative pair of rows: it holds the atoms of `a ⊗ b`, all of `S(a) × S(b)` when
/// `full` (no cancellation).
#[derive(Clone, Copy, Debug)]
pub(super) struct Rep {
    pub(super) a: usize,
    pub(super) b: usize,
    pub(super) full: bool,
}

/// Stage 1: the pairs that can never be dropped, and representatives holding exactly their
/// atoms. Each rule certifies a set in which every free atom of a member has two holders in
/// the set (said at each); a union of such sets is one too. The rules use the support `S(r)`
/// (the variables of `r`) and direction of each row on a pair, interned in O(T) hashing.
fn certify(rs: &Rows, pairs: &[(usize, usize)], k: usize) -> (Vec<bool>, Vec<Rep>) {
    let n_rows = rs.vars.len();
    let budget = CERTIFY_PER_INPUT.saturating_mul(k);
    let mut opp: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n_rows];
    for (e, &(a, b)) in pairs.iter().enumerate() {
        tick(1);
        opp[a].push((b, e));
        if a != b {
            opp[b].push((a, e));
        }
    }
    let mut sup_of: HashMap<Vec<usize>, usize> = HashMap::new();
    let mut dir_of: HashMap<Vec<(usize, Int)>, usize> = HashMap::new();
    let (mut sup, mut dir) = (vec![usize::MAX; n_rows], vec![usize::MAX; n_rows]);
    for r in 0..n_rows {
        if opp[r].is_empty() {
            continue;
        }
        tick(rs.len(r));
        let s: Vec<usize> = rs.vars[r].iter().map(|&(v, _)| v).collect();
        let next = sup_of.len();
        sup[r] = *sup_of.entry(s).or_insert(next);
        let d = rs.direction(r);
        let next = dir_of.len();
        dir[r] = *dir_of.entry(d).or_insert(next);
    }
    let mut cert = vec![false; pairs.len()];
    let mut reps: Vec<Rep> = Vec::new();
    // C1: at each row `p`, the pairs `{p, r}` whose rows `r` have one direction. Their
    // products are multiples of one another, so each atom of one is held by all of them: with
    // two or more, every atom has two holders. Two pairs of the same two rows have one
    // direction, so every repeated pair is certified here (P3 relies on this). The
    // representative is one of them, exact (its own atoms, with any cancellation).
    if !off(pass::C1) {
        for (p, opp_p) in opp.iter().enumerate() {
            if opp_p.len() < 2 {
                continue;
            }
            let mut by: Vec<(usize, usize, usize)> =
                opp_p.iter().map(|&(r, e)| (dir[r], e, r)).collect();
            tick(by.len());
            by.sort_unstable();
            for g in by.chunk_by(|x, y| x.0 == y.0) {
                if g.len() < 2 {
                    continue;
                }
                for &(_, e, _) in g {
                    cert[e] = true;
                }
                let r0 = g[0].2;
                reps.push(Rep {
                    a: p.min(r0),
                    b: p.max(r0),
                    full: false,
                });
            }
        }
    }
    // C2: at each row `p`, the pairs `{p, r}` whose rows `r` have one support. Each such
    // product holds only atoms of `S(p) × S(r)`; when two of them hold all of it (no
    // cancellation), every atom of every member has those two holders. A member holds all of
    // `S(p) × S(r)` when it and `p` have one sign each (C2a; no sum of two products of
    // like-signed numbers is zero), when `p` and `r` share at most one variable (C2b; a sum
    // `p_w r_z + p_z r_w` with both terms needs `w` and `z` both shared), or when no two
    // shared variables cancel (C2c, [`Rows::no_cancel`]). The representative is one such
    // member, full.
    // A group: its key (the supports of `p` and of its members, least first), the row `p`,
    // and its members, each the index of a pair and the pair's other row `r`.
    type Group = ((usize, usize), usize, Vec<(usize, usize)>);
    let mut pending: Vec<Group> = Vec::new();
    if !off(pass::C2) {
        for p in 0..n_rows {
            if opp[p].len() < 2 {
                continue;
            }
            let mut by: Vec<(usize, usize, usize)> =
                opp[p].iter().map(|&(r, e)| (sup[r], e, r)).collect();
            tick(by.len());
            by.sort_unstable();
            for g in by.chunk_by(|x, y| x.0 == y.0) {
                if g.len() < 2 || g.iter().all(|&(_, e, _)| cert[e]) {
                    continue;
                }
                tick(g.len());
                let mut monos = g.iter().filter(|&&(_, _, r)| rs.mono[r]).map(|x| x.2);
                if let (true, Some(m0), Some(_)) = (rs.mono[p], monos.next(), monos.next()) {
                    for &(_, e, _) in g {
                        cert[e] = true;
                    }
                    reps.push(Rep {
                        a: p,
                        b: m0,
                        full: true,
                    });
                } else {
                    let r0 = g[0].2;
                    let key = (sup[p].min(sup[r0]), sup[p].max(sup[r0]));
                    pending.push((key, p, g.iter().map(|&(_, e, r)| (e, r)).collect()));
                }
            }
        }
    }
    // C2b and C2c read the shared variables of `p` and each member, O(min(|p|, |r|) log)
    // each; a group whose rows have at most 2 variables on the shorter side always runs
    // (O(members)). The other groups run only when their cost, summed before any of them
    // runs, is at most `CERTIFY_PER_INPUT × k` (a static condition, part of Φ).
    pending.sort_unstable_by_key(|g| (g.0, g.1));
    let cost = |p: usize, r: usize| rs.len(p).min(rs.len(r));
    let heavy = (pending.iter())
        .filter(|g| cost(g.1, g.2[0].1) > 2)
        .map(|g| cost(g.1, g.2[0].1).saturating_mul(1 + g.2.len()))
        .fold(0usize, usize::saturating_add);
    tick(pending.len());
    for (_, p, members) in &pending {
        let p = *p;
        if cost(p, members[0].1) > 2 && heavy > budget {
            continue;
        }
        let full = if rs.shared(p, members[0].1) <= 1 {
            Some(members[0].1)
        } else {
            let mut whole = members.iter().map(|x| x.1).filter(|&r| rs.no_cancel(p, r));
            match (whole.next(), whole.next()) {
                (Some(r), Some(_)) => Some(r),
                _ => None,
            }
        };
        if let Some(r) = full {
            for &(e, _) in members {
                cert[e] = true;
            }
            reps.push(Rep {
                a: p,
                b: r,
                full: true,
            });
        }
    }
    // C3: a row `p` paired with rows of one variable `w` of `p` each, covering `S(p)`, every
    // `w` met once having `w * w` mentioned. The pair `{p, w}` holds the atoms `w * z`, `z` of
    // `p` (coefficients `p_z c_w`); an atom `w * z`, `w != z`, is held by the pairs of `w` and
    // of `z`, and `w * w` by two pairs of `w` or mentioned. Their atoms are all of
    // `S(p) × S(p)`, which `p ⊗ p` holds in full (`2 p_w p_z`, `p_w²`): the representative is
    // the virtual pair `{p, p}`, full. O(|opp[p]|) for each `p`.
    if !off(pass::C3) {
        for p in 0..n_rows {
            let lp = rs.len(p);
            if lp == 0 || opp[p].len() < lp {
                continue;
            }
            let mut count: HashMap<usize, usize> = HashMap::new();
            let mut single: Vec<usize> = Vec::new();
            for &(r, e) in &opp[p] {
                tick(1);
                if r != p && rs.len(r) == 1 {
                    let w = rs.vars[r][0].0;
                    if rs.coef(p, w).is_some() {
                        *count.entry(w).or_default() += 1;
                        single.push(e);
                    }
                }
            }
            let covered = count.len() == lp
                && (rs.vars[p].iter()).all(|&(w, _)| count[&w] >= 2 || rs.is_mentioned(w, w));
            if covered {
                for e in single {
                    cert[e] = true;
                }
                reps.push(Rep {
                    a: p,
                    b: p,
                    full: true,
                });
            }
        }
    }
    // For each row, the partner of a full representative on it with the most variables.
    let mut best: Vec<Option<usize>> = vec![None; n_rows];
    for rep in reps.iter().filter(|r| r.full) {
        for (p, s) in [(rep.a, rep.b), (rep.b, rep.a)] {
            tick(1);
            if best[p].is_none_or(|b| rs.len(b) < rs.len(s)) {
                best[p] = Some(s);
            }
        }
    }
    // C4: a pair `{p, r}` with `S(r)` inside `S(s)`, `{p, s}` a full representative. Its atoms
    // lie in `S(p) × S(r)`, inside `S(p) × S(s)`, all held by certified pairs, so each has two
    // holders once it is certified too; no representative is added. The test of a key
    // `(S(r), S(s))` reads `|r|` variables, once per key; keys with `|r| <= 2` always run, the
    // others only when their sum, computed before, is at most `CERTIFY_PER_INPUT × k` (part of
    // Φ).
    if !off(pass::C4) {
        let mut cand: Vec<((usize, usize), usize, usize, usize)> = Vec::new();
        for (e, &(a, b)) in pairs.iter().enumerate() {
            if cert[e] {
                continue;
            }
            for (p, r) in [(a, b), (b, a)] {
                tick(1);
                if let Some(s) = best[p] {
                    cand.push(((sup[r], sup[s]), r, s, e));
                }
            }
        }
        tick(cand.len());
        cand.sort_unstable();
        let heavy = (cand.chunk_by(|x, y| x.0 == y.0))
            .filter(|g| rs.len(g[0].1) > 2)
            .map(|g| rs.len(g[0].1))
            .fold(0usize, usize::saturating_add);
        for g in cand.chunk_by(|x, y| x.0 == y.0) {
            let (r, s) = (g[0].1, g[0].2);
            if rs.len(r) > 2 && heavy > budget {
                continue;
            }
            if (rs.vars[r].iter()).all(|&(v, _)| rs.coef(s, v).is_some()) {
                for x in g {
                    cert[x.3] = true;
                }
            }
        }
    }
    // C5: at each row `p`, the uncertified pairs `{p, q}` with `|q| <= |p|` whose products do
    // not cancel (so each holds all of `S(p) × S(q)`), cut to their 2-core: repeatedly remove
    // a `q` holding a variable `z` that no other `q` left holds and `best[p]` does not hold.
    // In the core each atom `w * z` (`w` of `p`, `z` of `q`) is held by another pair of the
    // core holding `z`, or by the certified pairs that `{p, best[p]}` stands for. Each pair
    // left is its own representative, full. Cost: Σ over the pairs of `min(|a|, |b|)` lookups
    // (the cancellation test and the core), the neighbor sum `W_nbr`; the pass runs only when
    // `W_nbr`, summed before, is at most `CERTIFY_PER_INPUT × k` (part of Φ).
    let w_nbr = (pairs.iter().enumerate())
        .filter(|&(e, _)| !cert[e])
        .map(|(_, &(a, b))| rs.len(a).min(rs.len(b)))
        .fold(0usize, usize::saturating_add);
    tick(pairs.len());
    if !off(pass::C5) && w_nbr <= budget {
        let mut nbrs: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n_rows];
        for (e, &(a, b)) in pairs.iter().enumerate() {
            if cert[e] {
                continue;
            }
            let (p, q) = if rs.len(a) > rs.len(b) || (rs.len(a) == rs.len(b) && a >= b) {
                (a, b)
            } else {
                (b, a)
            };
            if (rs.mono[a] && rs.mono[b]) || rs.no_cancel(a, b) {
                nbrs[p].push((q, e));
            }
        }
        for (p, ns) in nbrs.iter().enumerate() {
            if ns.is_empty() || (ns.len() < 2 && best[p].is_none()) {
                continue;
            }
            let mut count: HashMap<usize, usize> = HashMap::new();
            let mut on: HashMap<usize, Vec<usize>> = HashMap::new();
            for (i, &(q, _)) in ns.iter().enumerate() {
                for &(z, _) in &rs.vars[q] {
                    tick(1);
                    let c = count.entry(z).or_insert_with(|| {
                        if best[p].is_some_and(|s| rs.coef(s, z).is_some()) {
                            2
                        } else {
                            0
                        }
                    });
                    *c += 1;
                    on.entry(z).or_default().push(i);
                }
            }
            let mut alive = vec![true; ns.len()];
            let mut stack: Vec<usize> = (0..ns.len())
                .filter(|&i| (rs.vars[ns[i].0].iter()).any(|(z, _)| count[z] == 1))
                .collect();
            while let Some(i) = stack.pop() {
                if !alive[i] {
                    continue;
                }
                alive[i] = false;
                for &(z, _) in &rs.vars[ns[i].0] {
                    tick(1);
                    let c = count.get_mut(&z).expect("counted above");
                    *c -= 1;
                    if *c == 1 {
                        for &j in &on[&z] {
                            tick(1);
                            if alive[j] {
                                stack.push(j);
                            }
                        }
                    }
                }
            }
            for (i, &(q, e)) in ns.iter().enumerate() {
                if alive[i] {
                    cert[e] = true;
                    reps.push(Rep {
                        a: p,
                        b: q,
                        full: true,
                    });
                }
            }
        }
    }
    // One representative for each set of atoms: full ones by their two supports (one atom
    // set each), exact ones by their two directions (products that are multiples of each
    // other), and an exact one dropped when a full one has its supports (it holds a subset).
    let key = |a: usize, b: usize, of: &[usize]| (of[a].min(of[b]), of[a].max(of[b]));
    let mut full_keys: HashSet<(usize, usize)> = HashSet::new();
    let mut exact_keys: HashSet<(usize, usize)> = HashSet::new();
    let mut out: Vec<Rep> = Vec::new();
    for rep in reps.iter().filter(|r| r.full) {
        tick(1);
        if full_keys.insert(key(rep.a, rep.b, &sup)) {
            out.push(*rep);
        }
    }
    for rep in reps.iter().filter(|r| !r.full) {
        tick(1);
        if !full_keys.contains(&key(rep.a, rep.b, &sup))
            && exact_keys.insert(key(rep.a, rep.b, &dir))
        {
            out.push(*rep);
        }
    }
    (cert, out)
}

/// Stage 2 state: the live pairs (the uncertified pairs `0..n`, then the representatives,
/// which never go) by row, with the counts the rules read.
struct Peel<'r, 'a> {
    rs: &'r Rows<'a>,
    ends: Vec<(usize, usize)>,
    /// Each row's live pairs, as (other row, id); `pos[id]` is the place of `id` in the list
    /// of its first and of its second row, so a pair is removed in O(1) (swap-remove). With
    /// the lists of all pairs, dead ones read and skipped, Codex's family cost Θ(k²)
    /// (XR-presburger4 round 4, N-1: 61.1 times at 8 times; the live lists 9.59 times).
    opp: Vec<Vec<(usize, usize)>>,
    pos: Vec<[usize; 2]>,
    /// Σ |S(other)| over each row's live pairs.
    sum: Vec<usize>,
    /// Each row's uncertified pairs by the size of the other row, largest first; dead
    /// entries are popped when met.
    heap: Vec<BinaryHeap<(usize, usize)>>,
    live: Vec<bool>,
    /// The rows with a live pair holding each variable, counted.
    hcount: Vec<usize>,
    holders: Vec<Vec<usize>>,
    /// For a row that alone holds some variable `w`: the least `|mentioned(w)|` of those.
    min_m: Vec<Option<usize>>,
    /// For a row that alone holds some variable: the least such variable.
    uniq: Vec<Option<usize>>,
    queue: VecDeque<usize>,
    queued: Vec<bool>,
    /// Whether P4 runs (its static condition held).
    p4: bool,
    /// P4: for each variable `w`, at least `Σ sum[r]` over the rows `r` holding `w` (lowered
    /// only on the side of a removed pair whose other row is not shorter, so it stays an upper
    /// bound).
    over: Vec<usize>,
    /// P4: for each variable `w`, the uncertified pairs with `w` on their shorter row (both
    /// rows when of one length) and not on the other, by the other row's length, largest first.
    cand: Vec<BinaryHeap<(usize, usize)>>,
    vqueue: VecDeque<usize>,
    vqueued: Vec<bool>,
}

impl Peel<'_, '_> {
    fn wake(&mut self, r: usize) {
        if self.min_m[r].is_some() && !self.queued[r] {
            self.queued[r] = true;
            self.queue.push_back(r);
        }
    }

    /// Row `r` is now the only row with a live pair holding `v`.
    fn unique(&mut self, r: usize, v: usize) {
        let m = self.rs.mentioned[v].len();
        self.min_m[r] = Some(self.min_m[r].map_or(m, |x| x.min(m)));
        self.uniq[r] = Some(self.uniq[r].map_or(v, |u| u.min(v)));
        self.wake(r);
    }

    fn remove(&mut self, id: usize) {
        self.live[id] = false;
        let rs = self.rs;
        let (a, b) = self.ends[id];
        let sides = if a == b { 1 } else { 2 };
        for side in 0..sides {
            let (r, other) = if side == 0 { (a, b) } else { (b, a) };
            let i = self.pos[id][side];
            self.opp[r].swap_remove(i);
            if let Some(&(_, moved)) = self.opp[r].get(i) {
                let s = usize::from(self.ends[moved].0 != r);
                self.pos[moved][s] = i;
            }
            self.sum[r] -= rs.len(other);
            tick(1);
            if self.p4 && rs.len(other) >= rs.len(r) {
                for &(v, _) in &rs.vars[r] {
                    tick(1);
                    self.over[v] -= rs.len(other);
                    self.wake_var(v);
                }
            }
            if self.opp[r].is_empty() {
                self.row_dies(r);
            } else {
                self.wake(r);
            }
        }
    }

    fn wake_var(&mut self, v: usize) {
        if !self.cand[v].is_empty() && !self.vqueued[v] {
            self.vqueued[v] = true;
            self.vqueue.push_back(v);
        }
    }

    /// P4 on `w`: drops the pair of `cand[w]` with the longest other row while the test holds.
    /// Proof: let `{p, q}` be that pair, live and uncertified, `w` on `q` and not on `p`. Each
    /// `z` of `p` gives the atom `w * z` (`z != w`), whose coefficient in `p ⊗ q` is
    /// `q_w p_z + p_w q_z = q_w p_z`, not zero; distinct `z`, distinct atoms. Another live pair
    /// or representative holding `w * z` has a row holding `w` whose partner holds `z` (its
    /// coefficient `s_w t_z + s_z t_w` is zero otherwise), so the `z` blocked, or mentioned
    /// with `w`, are at most `|mentioned(w)| + Σ_{s ∋ w} sum[s] - |p|`, the `- |p|` being the
    /// pair's own part of `sum[q]`; and `over[w]` is at least that sum. When
    /// `2 |p| > |mentioned(w)| + over[w]`, some `z` of `p` is not blocked: `w * z` is free and
    /// held by `{p, q}` alone, and the pair goes. The test is monotone in `|p|`, so only the
    /// longest is tried; `w` is tried again when `over[w]` falls.
    fn p4(&mut self, w: usize) {
        let m = self.rs.mentioned[w].len();
        while let Some(&(len, id)) = self.cand[w].peek() {
            tick(1);
            if !self.live[id] {
                self.cand[w].pop();
            } else if 2 * len > m + self.over[w] {
                self.cand[w].pop();
                self.remove(id);
            } else {
                break;
            }
        }
    }

    /// Row `r` has no live pair left: each of its variables loses a holder. A variable whose
    /// count falls to one has its last row found by a walk of its holders, once in all.
    fn row_dies(&mut self, r: usize) {
        let rs = self.rs;
        for &(v, _) in &rs.vars[r] {
            tick(1);
            self.hcount[v] -= 1;
            if self.hcount[v] == 1 {
                let opp = &self.opp;
                let s = (self.holders[v].iter().copied())
                    .find(|&s| {
                        tick(1);
                        !opp[s].is_empty()
                    })
                    .expect("one row with a live pair holds it");
                self.unique(s, v);
            }
        }
    }
}

/// Stages 2 and 3 on the uncertified `pairs` beside the representatives `reps`: which of
/// `pairs` are kept, or the refusal Φ ([`kept`]).
pub(super) fn settle(
    rs: &Rows,
    pairs: &[(usize, usize)],
    reps: &[Rep],
    k: usize,
) -> Result<Vec<bool>, Refusal> {
    let n = pairs.len();
    let n_rows = rs.vars.len();
    let n_vars = rs.mentioned.len();
    let ends: Vec<(usize, usize)> = (pairs.iter().copied())
        .chain(reps.iter().map(|r| (r.a, r.b)))
        .collect();
    let mut peel = Peel {
        rs,
        opp: vec![Vec::new(); n_rows],
        pos: vec![[0; 2]; ends.len()],
        sum: vec![0; n_rows],
        heap: vec![BinaryHeap::new(); n_rows],
        live: vec![true; ends.len()],
        hcount: vec![0; n_vars],
        holders: vec![Vec::new(); n_vars],
        min_m: vec![None; n_rows],
        uniq: vec![None; n_rows],
        queue: VecDeque::new(),
        queued: vec![false; n_rows],
        p4: false,
        over: vec![0; n_vars],
        cand: vec![BinaryHeap::new(); n_vars],
        vqueue: VecDeque::new(),
        vqueued: vec![false; n_vars],
        ends,
    };
    for id in 0..peel.ends.len() {
        tick(1);
        let (a, b) = peel.ends[id];
        let sides = if a == b { 1 } else { 2 };
        for side in 0..sides {
            let (r, other) = if side == 0 { (a, b) } else { (b, a) };
            peel.pos[id][side] = peel.opp[r].len();
            peel.opp[r].push((other, id));
            peel.sum[r] += rs.len(other);
            if id < n {
                peel.heap[r].push((rs.len(other), id));
            }
        }
    }
    for r in 0..n_rows {
        if !peel.opp[r].is_empty() {
            for &(v, _) in &rs.vars[r] {
                tick(1);
                peel.hcount[v] += 1;
                peel.holders[v].push(r);
            }
        }
    }
    for v in 0..n_vars {
        tick(1);
        if peel.hcount[v] == 1 {
            let r = peel.holders[v][0];
            peel.unique(r, v);
        }
    }
    // P4's static condition (part of Φ): its work is O(W4 log) with
    // `W4 = Σ min(|a|, |b|)` over the uncertified pairs of two rows (the entries of `cand`,
    // and the lowerings of `over`, each on the side of a removed pair that is not longer),
    // and it runs only when `W4 <= CERTIFY_PER_INPUT × k`, summed before in O(P).
    let w4 = (pairs.iter())
        .filter(|&&(a, b)| a != b)
        .map(|&(a, b)| rs.len(a).min(rs.len(b)))
        .fold(0usize, usize::saturating_add);
    tick(n);
    if !off(pass::P4) && w4 <= CERTIFY_PER_INPUT.saturating_mul(k) {
        peel.p4 = true;
        for r in 0..n_rows {
            for &(v, _) in &rs.vars[r] {
                tick(1);
                peel.over[v] += peel.sum[r];
            }
        }
        for (id, &(a, b)) in pairs.iter().enumerate() {
            if a == b {
                continue;
            }
            for (q, p) in [(a, b), (b, a)] {
                if rs.len(q) > rs.len(p) {
                    continue;
                }
                for &(w, _) in &rs.vars[q] {
                    if rs.coef(p, w).is_none() {
                        peel.cand[w].push((rs.len(p), id));
                    }
                }
            }
        }
        for v in 0..n_vars {
            peel.wake_var(v);
        }
    }
    // P0: a pair of two rows of one variable each holds the one atom `w * z`; when it is
    // mentioned the pair never goes and blocks nothing that matters, so it is kept and taken
    // out of the live lists.
    let mut kept = vec![false; n];
    for (id, &(a, b)) in pairs.iter().enumerate() {
        tick(1);
        if rs.len(a) == 1 && rs.len(b) == 1 && rs.is_mentioned(rs.vars[a][0].0, rs.vars[b][0].0) {
            kept[id] = true;
            peel.remove(id);
        }
    }
    // P3: `w` held only by `a` and `z` only by `b` (rows with a live pair), `w * z` free and
    // held by `{a, b}`: any live pair holding `w * z` has a row holding `w` and one holding
    // `z`, so its rows are `a` and `b`. No other uncertified pair has those rows (repeated
    // pairs are certified by C1); a representative with those rows may hold it, and then the
    // pair is not dropped. Without that test, a residual `{p, p}` beside C3's virtual
    // representative `{p, p}` was dropped while every one of its atoms was blocked (the soundness
    // fault in Neon's P3, XR-presburger4 presburger5 handoff; test
    // `tests::p3_keeps_a_pair_whose_rows_a_representative_has`). For `a = b` the atom is
    // `w * w`. One pass, O(P log).
    let rep_rows: HashSet<(usize, usize)> = (reps.iter())
        .map(|r| (r.a.min(r.b), r.a.max(r.b)))
        .collect();
    if !off(pass::P3) {
        for (id, &(a, b)) in pairs.iter().enumerate() {
            tick(1);
            if !peel.live[id] {
                continue;
            }
            let (Some(w), Some(z)) = (peel.uniq[a], peel.uniq[b]) else {
                continue;
            };
            if !off(pass::P3_GUARD) && rep_rows.contains(&(a.min(b), a.max(b))) {
                continue;
            }
            if !rs.is_mentioned(w, z) && rs.holds(a, b, w, z) {
                peel.remove(id);
            }
        }
    }
    // P2: a row `r` alone holds `w` (among rows with a live pair). Each live pair `{r, p}`
    // holds `w * z` for every `z` of `p` (coefficient `r_w p_z`, as `p` does not hold `w`;
    // `2 r_w r_z` or `r_w²` for `p = r`), and any other live pair holding `w * z` is on `r`
    // (it needs a row holding `w`), so it is another `{r, s}` with `z` on `s`. The `z` of `p`
    // so blocked, or mentioned with `w`, are at most `|mentioned(w)| + Σ_{s != p} |S(s)|`; when
    // `|S(p)|` is more, some atom `w * z` is free and held by `{r, p}` alone, and it goes.
    // The test is monotone in `|S(p)|`, so only the largest partner is tried. A row is tried
    // again when its sum falls or a new variable becomes its own: O(log) a try, and each try
    // is caused by a drop or by a variable, O((P + T) log) in all.
    // P4 ([`Peel::p4`]): P2 for a variable `w` held by several rows, against the sum of the
    // partners of all of them; tried when P2's queue is empty, each drop of either waking the
    // other. Without it a sum `b0 + ... + b{k-1}` paired with each `x{i} - b{i} - c{i}` and
    // each `c{i}`, whose `x{i}` and `c{i}` are on other rows too, left its 2k pairs, Θ(k²)
    // terms, to stage 3, and was refused (l01_38 bench
    // `products_whose_factors_one_sum_on_the_factors_joins_scale`; `tests::a_sum_on_the_factors_scale`).
    loop {
        while let Some(r) = peel.queue.pop_front() {
            peel.queued[r] = false;
            let Some(m) = peel.min_m[r] else {
                continue;
            };
            if off(pass::P2) {
                continue;
            }
            while let Some(&(len, id)) = peel.heap[r].peek() {
                tick(1);
                if !peel.live[id] {
                    peel.heap[r].pop();
                } else if 2 * len > m + peel.sum[r] {
                    peel.heap[r].pop();
                    peel.remove(id);
                } else {
                    break;
                }
            }
        }
        let Some(w) = peel.vqueue.pop_front() else {
            break;
        };
        peel.vqueued[w] = false;
        peel.p4(w);
    }
    // Stage 3 (Solver A of NEON.md): expand the pairs left, the core `E`, and peel them by
    // their free atoms' holders, an atom held by a representative never freeing one.
    let core: Vec<usize> = (0..n).filter(|&i| peel.live[i]).collect();
    let limit = EXPAND_PER_INPUT.saturating_mul(k);
    let size = |(a, b): (usize, usize)| rs.len(a).saturating_mul(rs.len(b));
    let w_exp = (core.iter())
        .map(|&i| size(pairs[i]))
        .fold(0usize, usize::saturating_add);
    tick(core.len());
    if w_exp > limit {
        return Err(Refusal::Products);
    }
    let mut atom_id: HashMap<(usize, usize), usize> = HashMap::new();
    let mut atoms: Vec<(usize, usize)> = Vec::new();
    let mut count: Vec<usize> = Vec::new();
    let mut holders: Vec<Vec<usize>> = Vec::new();
    let mut pair_atoms: Vec<Vec<usize>> = Vec::with_capacity(core.len());
    for (ci, &i) in core.iter().enumerate() {
        let mut mine = Vec::new();
        for (t, c) in expand(rs, pairs[i]) {
            if c.is_zero() || rs.is_mentioned(t.0, t.1) {
                continue;
            }
            let next = atoms.len();
            let id = *atom_id.entry(t).or_insert(next);
            if id == next {
                atoms.push(t);
                count.push(0);
                holders.push(Vec::new());
            }
            count[id] += 1;
            holders[id].push(ci);
            mine.push(id);
        }
        pair_atoms.push(mine);
    }
    let mut alive = vec![true; core.len()];
    if count.contains(&1) {
        // Which atoms the representatives hold: by `Q_rep` lookups (for each atom, the
        // representatives on the rows holding the variable of the two with fewer, each a
        // test of its coefficient; any representative holding `w * z` has a row holding `w`
        // and one holding `z`, so it is among them) or by expanding every representative
        // (`W_rep` terms), whichever is less; both sums are computed before, and their least
        // added to `W_exp` must be within the limit (Φ).
        let mut reps_on: Vec<Vec<usize>> = vec![Vec::new(); n_rows];
        for (j, rep) in reps.iter().enumerate() {
            tick(1);
            reps_on[rep.a].push(j);
            if rep.b != rep.a {
                reps_on[rep.b].push(j);
            }
        }
        let mut rep_rows_of: Vec<Vec<usize>> = vec![Vec::new(); n_vars];
        let mut rw = vec![0usize; n_vars];
        for (r, on) in reps_on.iter().enumerate() {
            if !on.is_empty() {
                for &(v, _) in &rs.vars[r] {
                    tick(1);
                    rep_rows_of[v].push(r);
                    rw[v] = rw[v].saturating_add(on.len());
                }
            }
        }
        let q_rep = (atoms.iter())
            .map(|&(w, z)| rw[w].min(rw[z]))
            .fold(0usize, usize::saturating_add);
        let w_rep = (reps.iter())
            .map(|r| size((r.a, r.b)))
            .fold(0usize, usize::saturating_add);
        tick(atoms.len() + reps.len());
        if w_exp.saturating_add(q_rep.min(w_rep)) > limit {
            return Err(Refusal::Products);
        }
        let blocked: Vec<bool> = if q_rep <= w_rep {
            (atoms.iter())
                .map(|&(w, z)| {
                    let v = if rw[w] <= rw[z] { w } else { z };
                    (rep_rows_of[v].iter()).any(|&r| {
                        (reps_on[r].iter()).any(|&j| {
                            tick(1);
                            rs.holds(reps[j].a, reps[j].b, w, z)
                        })
                    })
                })
                .collect()
        } else {
            let mut held: HashSet<(usize, usize)> = HashSet::new();
            for rep in reps {
                for (t, c) in expand(rs, (rep.a, rep.b)) {
                    if !c.is_zero() {
                        held.insert(t);
                    }
                }
            }
            atoms.iter().map(|t| held.contains(t)).collect()
        };
        let mut queue: Vec<usize> = (0..atoms.len()).filter(|&t| count[t] == 1).collect();
        while let Some(t) = queue.pop() {
            tick(1);
            if count[t] != 1 || blocked[t] {
                continue;
            }
            let Some(ci) = (holders[t].iter().copied()).find(|&ci| {
                tick(1);
                alive[ci]
            }) else {
                continue;
            };
            alive[ci] = false;
            for &t2 in &pair_atoms[ci] {
                tick(1);
                count[t2] -= 1;
                if count[t2] == 1 {
                    queue.push(t2);
                }
            }
        }
    }
    for (ci, &i) in core.iter().enumerate() {
        kept[i] = alive[ci];
    }
    Ok(kept)
}

/// The atoms of `a ⊗ b` with their coefficients, in order of atom: `|a| × |b|` terms summed
/// by atom `{x, y}` (each unordered pair once).
fn expand(rs: &Rows, (a, b): (usize, usize)) -> Vec<((usize, usize), Int)> {
    let mut local: HashMap<(usize, usize), Int> = HashMap::new();
    for &(x, cx) in &rs.vars[a] {
        for &(y, cy) in &rs.vars[b] {
            tick(1);
            let t = cx * cy;
            local
                .entry((x.min(y), x.max(y)))
                .and_modify(|c| *c = &*c + &t)
                .or_insert(t);
        }
    }
    let mut out: Vec<((usize, usize), Int)> = local.into_iter().collect();
    out.sort_unstable_by_key(|x| x.0);
    out
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::collections::HashMap;

    use super::super::Refusal;
    use super::super::formula::{Atom, Factor, Lin};
    use super::{ANSWERED, INPUT, OFF, Rep, Rows, WORK, kept, pass, settle};

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

    /// [`row`] with owned names.
    fn row_of(constant: i64, terms: &[(i64, String)]) -> Lin {
        let terms: Vec<(i64, &str)> = terms.iter().map(|(c, s)| (*c, s.as_str())).collect();
        row(constant, &terms)
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
        assert_eq!(kept(&rows, &pairs, &partners).expect("answered"), [true; 4]);
        assert_eq!(expanded(&rows, &pairs, &partners), [true; 4]);
        // Without the product of `-a - b - d` and `-c`, `(1 - a - d) * (1 + a - b + c)` alone
        // holds `c * d`, and then `(b - a) * (-a - b - d)` alone holds `a * d`.
        assert_eq!(
            kept(&rows, &[(0, 1), (2, 3), (4, 4)], &partners).expect("answered"),
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
        assert_eq!(kept(&rows, &[(0, 1)], &both).expect("answered"), [true]);
        let one = partners_of(&[("x", "x")]);
        assert_eq!(kept(&rows, &[(0, 1)], &one).expect("answered"), [false]);
    }

    #[test]
    fn the_pairs_kept_are_those_left_by_expanding_every_product() {
        // Random rows over a few symbols, random pairs and random mentioned products, the
        // pairs given in a random order: `kept` agrees with `expanded`, so its answer does not
        // depend on the order of the pairs. Each case is also run with each pass of stages 1
        // and 2 turned off alone (except the guard of P3, which is a repair): each only
        // certifies or drops fewer pairs, never changing the answer.
        let mut rng = crate::seed::Rng::new(0x2545_f491_4f6c_dd1d);
        let mut next = move |n: usize| rng.below(n);
        let names = ["a", "b", "c", "d", "e", "f"];
        let mut dropped = 0;
        let passes = [
            0,
            pass::C1,
            pass::C2,
            pass::C3,
            pass::C4,
            pass::C5,
            pass::P2,
            pass::P3,
        ];
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
            for bits in passes {
                OFF.with(|o| o.set(bits));
                let got = kept(&rows, &pairs, &partners);
                OFF.with(|o| o.set(0));
                assert_eq!(
                    got.expect("answered"),
                    want,
                    "case {case}, off {bits}: rows {rows:?}, pairs {pairs:?}, products {products:?}"
                );
            }
            dropped += want.iter().filter(|&&k| !k).count();
        }
        assert!(dropped > 0, "no case dropped a pair");
    }

    /// The fault in Neon's P3 (presburger5 handoff, order 3): `p = a + b + c` paired with
    /// itself, left uncertified, beside C3's virtual representative `{p, p}`, which stands for
    /// certified pairs holding every atom of `S(p) × S(p)`. `a` is held by `p` alone (the rows
    /// of one variable are on certified pairs only), so P3 took `a * a` as held by `{p, p}`
    /// alone and dropped it, though the representative holds it. The truth, from the pairs
    /// the representative stands for: `{p, p}` and the six pairs of `p` with two rows on each
    /// of `a`, `b`, `c` are all kept (`expanded`). Red without the guard (the second half of
    /// the test turns it off and gets the wrong answer).
    #[test]
    fn p3_keeps_a_pair_whose_rows_a_representative_has() {
        let p = row(0, &[(1, "a"), (1, "b"), (1, "c")]);
        let mut all = vec![p];
        for v in ["a", "a", "b", "b", "c", "c"] {
            all.push(row(0, &[(1, v)]));
        }
        let rows: Vec<&Lin> = all.iter().collect();
        let none = HashMap::new();
        let mut pairs: Vec<(usize, usize)> = vec![(0, 0)];
        pairs.extend((1..7).map(|r| (0, r)));
        assert_eq!(expanded(&rows, &pairs, &none), [true; 7]);
        assert_eq!(kept(&rows, &pairs, &none).expect("answered"), [true; 7]);
        let rs = Rows::new(&rows, &none);
        let virtual_rep = [Rep {
            a: 0,
            b: 0,
            full: true,
        }];
        let settled = settle(&rs, &[(0, 0)], &virtual_rep, 100);
        assert_eq!(settled, Ok(vec![true]), "with the guard");
        OFF.with(|o| o.set(pass::P3_GUARD));
        let unguarded = settle(&rs, &[(0, 0)], &virtual_rep, 100);
        OFF.with(|o| o.set(0));
        assert_eq!(unguarded, Ok(vec![false]), "the fault, without the guard");
    }

    /// A family of inputs of [`kept`]: rows, pairs and the products mentioned.
    struct Family {
        rows: Vec<Lin>,
        pairs: Vec<(usize, usize)>,
        products: Vec<(String, String)>,
    }

    /// The work of [`kept`] on `f` with the passes `bits` off ([`OFF`]), its answer and its
    /// input; checked against `expanded` when `check`.
    fn run(f: &Family, bits: u32, check: bool) -> (u64, Result<Vec<bool>, Refusal>, u64) {
        let products: Vec<(&str, &str)> = (f.products.iter())
            .map(|(u, v)| (u.as_str(), v.as_str()))
            .collect();
        let partners = partners_of(&products);
        let rows: Vec<&Lin> = f.rows.iter().collect();
        OFF.with(|o| o.set(bits));
        WORK.with(|w| w.set(0));
        let got = kept(&rows, &f.pairs, &partners);
        let work = WORK.with(Cell::get);
        OFF.with(|o| o.set(0));
        if check {
            let want = expanded(&rows, &f.pairs, &partners);
            assert_eq!(got.as_ref(), Ok(&want), "rows {rows:?}");
        }
        (work, got, INPUT.with(Cell::get))
    }

    /// Checks `build` against `expanded` at small sizes, then measures the work at `n` and
    /// `8 n`: both answered, the input about 8 times, the work at most 10.5 times (the scaling
    /// limit; the ratio is printed and the preferred 9 named in the report). The negative
    /// control turns off `control` and must refuse at `8 n` (its stage is what keeps the
    /// family outside Φ). The answers at `8 n` must satisfy `want`.
    fn scale(
        name: &str,
        build: impl Fn(usize) -> Family,
        n: usize,
        control: u32,
        want: impl Fn(&[bool]) -> bool,
    ) -> (u64, u64) {
        for small in [1, 2, 3, 5, 8] {
            // `run` with `check` asserts the answer equals `expanded`'s.
            let _ = run(&build(small), 0, true);
        }
        let (a, got_a, in_a) = run(&build(n), 0, false);
        let (b, got_b, in_b) = run(&build(8 * n), 0, false);
        assert!(got_a.is_ok(), "{name}: refused at n = {n}");
        let got_b = got_b.expect("answered at 8 n");
        assert!(want(&got_b), "{name}: unexpected answer at 8 n");
        eprintln!(
            "{name}: work {a} -> {b} ({:.2}x), input {in_a} -> {in_b} ({:.2}x)",
            b as f64 / a as f64,
            in_b as f64 / in_a as f64
        );
        assert!(in_b >= 7 * in_a, "{name}: input {in_a} -> {in_b}");
        assert!(2 * b <= 21 * a, "{name}: work {a} -> {b}");
        if control != 0 {
            let (_, ctl, _) = run(&build(8 * n), control, false);
            assert_eq!(ctl, Err(Refusal::Products), "{name}: negative control");
        }
        (a, b)
    }

    /// Σ over the pairs of the rows holding each variable of the pair's two rows (all rows,
    /// on a pair or not): the walk of a method that reads every holder of each variable of a
    /// pair, the negative control of the families no pass of this method is needed for.
    fn holder_walk(f: &Family) -> u64 {
        let mut on: HashMap<String, u64> = HashMap::new();
        for r in &f.rows {
            for a in r.terms.keys() {
                *on.entry(format!("{a:?}")).or_default() += 1;
            }
        }
        (f.pairs.iter())
            .flat_map(|&(a, b)| f.rows[a].terms.iter().chain(&f.rows[b].terms))
            .map(|(a, _)| on[&format!("{a:?}")])
            .sum()
    }

    /// A sum `w0 + ... + w{k-1} >= 0` paired with `c{j} >= 0` for each product
    /// `w{j} * c{j}`, and `y - w{j} >= 0` paired with `c{j}` and with `x - 1 >= 0` for `x * y`.
    fn sum_beside(k: usize) -> Family {
        let sum: Vec<(i64, String)> = (0..k).map(|j| (1, format!("w{j}"))).collect();
        let mut rows = vec![row(-1, &[(1, "x")]), row_of(0, &sum)];
        let mut pairs = Vec::new();
        let mut products = vec![("x".to_string(), "y".to_string())];
        for j in 0..k {
            let a = rows.len();
            rows.push(row_of(0, &[(1, "y".into()), (-1, format!("w{j}"))]));
            rows.push(row_of(0, &[(1, format!("c{j}"))]));
            pairs.extend([(0, a), (1, a + 1), (a, a + 1)]);
            products.push((format!("w{j}"), format!("c{j}")));
        }
        Family {
            rows,
            pairs,
            products,
        }
    }

    /// The rows of `products_whose_factors_one_sum_on_the_factors_joins_scale` (l01_38 ladder
    /// bench): for each `i` the rows `x{i} - z{i}`, `x{i} + b{i} - 10`, `x{i} - b{i} - c{i} - 4`,
    /// `c{i}` and `y{i} - z{i}`, the products `x{i} * y{i}` and `b{i} * c{i}`, and the sum
    /// `S = b0 + ... + b{k-1} + 1000000 k` on every factor `b{i}`.
    fn sum_on_factors(k: usize) -> Family {
        let sum: Vec<(i64, String)> = (0..k).map(|i| (1, format!("b{i}"))).collect();
        let mut rows = vec![row_of(1_000_000 * k as i64, &sum)];
        let mut pairs = Vec::new();
        let mut products = Vec::new();
        for i in 0..k {
            let (x, y, z, b, c) = (
                format!("x{i}"),
                format!("y{i}"),
                format!("z{i}"),
                format!("b{i}"),
                format!("c{i}"),
            );
            let a = rows.len();
            rows.push(row_of(0, &[(1, x.clone()), (-1, z.clone())]));
            rows.push(row_of(-10, &[(1, x.clone()), (1, b.clone())]));
            rows.push(row_of(
                -4,
                &[(1, x.clone()), (-1, b.clone()), (-1, c.clone())],
            ));
            rows.push(row_of(0, &[(1, c.clone())]));
            rows.push(row_of(0, &[(1, y.clone()), (-1, z)]));
            let (xz, xb, l, cc, yz) = (a, a + 1, a + 2, a + 3, a + 4);
            pairs.extend([(xz, yz), (xb, yz), (l, yz)]);
            pairs.extend([(xb, l), (xb, cc), (l, l), (l, cc), (0, l), (0, cc)]);
            products.extend([(x, y), (b, c)]);
        }
        Family {
            rows,
            pairs,
            products,
        }
    }

    /// [`sum_on_factors`]: no row alone holds a variable, so P2 drops none of the sum's pairs;
    /// P4 drops `{S, x{i} - b{i} - c{i} - 4}` on `x{i}` (`b{l} * x{i}` is its own), then
    /// `{S, c{i}}` on `c{i}` (`b{l} * c{i}` is then its own). Control: without P4 the 2k
    /// pairs of the sum reach stage 3, about 4k² terms, refused.
    #[test]
    fn a_sum_on_the_factors_scale() {
        scale("sum_on_factors", sum_on_factors, 120, pass::P4, |_| true);
    }

    /// XR-presburger4 round 1 A4 (Claude F3, F4) and NEON.md: P2 drops `{S, c{j}}` (`c{j}`
    /// is `c{j}`'s alone and the sum is longer than the rest of the row's partners), then the
    /// short pairs left are peeled in stage 3. Control: without P2, P3 and P4 (P4 alone also
    /// drops `{S, c{j}}`, on `c{j}`) the k pairs of the sum reach stage 3, k² terms, refused.
    #[test]
    fn a_sum_beside_its_products_scale() {
        scale(
            "sum_beside",
            sum_beside,
            250,
            pass::P2 | pass::P3 | pass::P4,
            |k| k.iter().all(|&x| !x),
        );
    }

    /// The row `x >= 0` and the sum `x + z0 + ... + z{k-1} >= 0`, each paired with the k rows
    /// `y - w{j} >= 0`, `x * y` mentioned (`products_whose_premise_holds_k_atoms_in_no_product`).
    fn premise_k(k: usize) -> Family {
        let mut sum: Vec<(i64, String)> = vec![(1, "x".into())];
        sum.extend((0..k).map(|j| (1, format!("z{j}"))));
        let mut rows = vec![row(0, &[(1, "x")]), row_of(0, &sum)];
        let mut pairs = Vec::new();
        for j in 0..k {
            let a = rows.len();
            rows.push(row_of(0, &[(1, "y".into()), (-1, format!("w{j}"))]));
            pairs.extend([(0, a), (1, a)]);
        }
        Family {
            rows,
            pairs,
            products: vec![("x".into(), "y".into())],
        }
    }

    /// P3 drops `{S, y - w{j}}` (`z0` the sum's own, `w{j}` the row's), and P2 or stage 3
    /// the rest. Control: without P2, P3 and P4 (P4 alone also drops them) the k pairs of the
    /// sum reach stage 3, refused.
    #[test]
    fn a_premise_holding_k_atoms_in_no_product_scale() {
        scale(
            "premise_k",
            premise_k,
            250,
            pass::P2 | pass::P3 | pass::P4,
            |_| true,
        );
    }

    /// Round 2, L1 (Codex): `x - 1`, `y`, the k rows `z + u{j}` on no pair, and the k rows
    /// `y + (j + 1) z`, each paired with `x - 1` for `x * y`, and `{x - 1, y}`.
    fn l1(k: usize) -> Family {
        let mut rows = vec![row(-1, &[(1, "x")]), row(0, &[(1, "y")])];
        let mut pairs = vec![(0, 1)];
        for j in 0..k {
            rows.push(row_of(0, &[(1, "z".into()), (1, format!("u{j}"))]));
        }
        for j in 0..k {
            pairs.push((0, rows.len()));
            rows.push(row(0, &[(1, "y"), (j as i64 + 1, "z")]));
        }
        Family {
            rows,
            pairs,
            products: vec![("x".into(), "y".into())],
        }
    }

    /// C2a certifies the k pairs `{x - 1, y + (j + 1) z}`, P0 keeps `{x - 1, y}`; the rows on
    /// no pair are never walked. Control: the walk of every holder of each variable of each
    /// pair (as the method of round 2 did) grows past 10.5 times.
    #[test]
    fn a_premise_of_rows_on_no_pair_scale() {
        let (a, b) = scale("l1", l1, 250, 0, |k| k.iter().all(|&x| x));
        let (wa, wb) = (holder_walk(&l1(250)), holder_walk(&l1(2000)));
        eprintln!("l1 control: holder walk {wa} -> {wb}, work {a} -> {b}");
        assert!(2 * wb > 21 * wa, "control {wa} -> {wb}");
    }

    /// NEON.md's cascade chain: `y`, the rows `x + u{i-1} + u{i}` (`u{-1}` = `z0`) each
    /// paired with `y`, `u{k-1} * y` mentioned, and the bystanders `{x + a{j}, y + c{j}}` with
    /// `x * c{j}`, `a{j} * y`, `a{j} * c{j}` mentioned. The chain goes one pair at a time.
    fn cascade(k: usize) -> Family {
        let u = |i: usize| {
            if i == 0 {
                "z0".to_string()
            } else {
                format!("u{}", i - 1)
            }
        };
        let mut rows = vec![row(0, &[(1, "y")])];
        let mut pairs = Vec::new();
        let mut products = vec![(format!("u{}", k - 1), "y".to_string())];
        for i in 0..k {
            pairs.push((rows.len(), 0));
            rows.push(row_of(
                0,
                &[(1, "x".into()), (1, u(i)), (1, format!("u{i}"))],
            ));
        }
        for j in 0..k {
            let (a, c) = (format!("a{j}"), format!("c{j}"));
            pairs.push((rows.len(), rows.len() + 1));
            rows.push(row_of(0, &[(1, "x".into()), (1, a.clone())]));
            rows.push(row_of(0, &[(1, "y".into()), (1, c.clone())]));
            products.extend([("x".into(), c.clone()), (a.clone(), "y".into()), (a, c)]);
        }
        Family {
            rows,
            pairs,
            products,
        }
    }

    /// P2 drops the chain one pair at a time as each `u{i}` becomes one row's own; the
    /// bystanders stay (stage 3: `x * y` has k holders). Control: the holder walk (the
    /// per-pair watches of round 3 measured 63.78 times).
    #[test]
    fn a_cascade_chain_scale() {
        let (a, b) = scale("cascade", cascade, 250, 0, |k| {
            let n = k.len() / 2;
            k[..n].iter().all(|&x| !x) && k[n..].iter().all(|&x| x)
        });
        let (wa, wb) = (holder_walk(&cascade(250)), holder_walk(&cascade(2000)));
        eprintln!("cascade control: holder walk {wa} -> {wb}, work {a} -> {b}");
        assert!(2 * wb > 21 * wa, "control {wa} -> {wb}");
    }

    /// Gemini's disjoint gadgets (round 4, verify_r4.rs `build_cheap_refused`), one copy
    /// scaled in `l`: `p = u0 + ... + u{l+1}`, `q1 = v0 + ... + v{l-1} + a`,
    /// `q2 = v0 + 2 v1 + ... + l v{l-1}`, two rows `a` and the rows `d1`, `d2`; the pairs
    /// `{p, q1}`, `{p, q2}`, `{a, d1}`, `{a, d2}`; nothing mentioned. None is kept.
    fn gadgets(l: usize) -> Family {
        let p: Vec<(i64, String)> = (0..l + 2).map(|i| (1, format!("u{i}"))).collect();
        let mut q1: Vec<(i64, String)> = (0..l).map(|j| (1, format!("v{j}"))).collect();
        q1.push((1, "a".into()));
        let q2: Vec<(i64, String)> = (0..l).map(|j| (j as i64 + 1, format!("v{j}"))).collect();
        let rows = vec![
            row_of(0, &p),
            row_of(0, &q1),
            row_of(0, &q2),
            row(0, &[(1, "a")]),
            row(0, &[(1, "d1")]),
            row(0, &[(1, "a")]),
            row(0, &[(1, "d2")]),
        ];
        Family {
            rows,
            pairs: vec![(0, 1), (0, 2), (3, 4), (5, 6)],
            products: Vec::new(),
        }
    }

    /// P3 or P2 drops every pair (each `u{i}` is `p`'s own). Control: without P2, P3 and P4
    /// (P4 alone also drops them) the two long products reach stage 3, about 2 l² terms,
    /// refused.
    #[test]
    fn disjoint_gadgets_scale() {
        scale(
            "gadgets",
            gadgets,
            250,
            pass::P2 | pass::P3 | pass::P4,
            |k| k.iter().all(|&x| !x),
        );
    }

    /// Claude's three-row family (round 4): the rows `Σ_i (1 + j (i + 1)) v{i}`, j = 0, 1, 2,
    /// over t variables, and all six pairs of them, with themselves too. No two are parallel
    /// and all have one sign, so C2a certifies all six at once.
    fn three_rows(t: usize) -> Family {
        let rows: Vec<Lin> = (0..3)
            .map(|j| {
                let terms: Vec<(i64, String)> = (0..t)
                    .map(|i| (1 + j * (i as i64 + 1), format!("v{i}")))
                    .collect();
                row_of(0, &terms)
            })
            .collect();
        Family {
            rows,
            pairs: vec![(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)],
            products: Vec::new(),
        }
    }

    /// C2a keeps all six. Control: without C2 and C5, the six products of t² terms each
    /// reach stage 3, refused.
    #[test]
    fn three_rows_on_one_support_scale() {
        scale("three_rows", three_rows, 100, pass::C2 | pass::C5, |k| {
            k.iter().all(|&x| x)
        });
    }

    /// Codex's adversarial and live-list family (round 4, neon_live_control.rs `stale`): the
    /// sum `w0 + ... + w{n-1}` paired with n rows `a{i}`, the last `a{n-1}` mentioned with
    /// every `w{i}`. Only the last pair is kept.
    fn stale(n: usize) -> Family {
        let sum: Vec<(i64, String)> = (0..n).map(|i| (1, format!("w{i}"))).collect();
        let mut rows = vec![row_of(0, &sum)];
        rows.extend((0..n).map(|i| row_of(0, &[(1, format!("a{i}"))])));
        let last = format!("a{}", n - 1);
        Family {
            rows,
            pairs: (1..=n).map(|i| (0, i)).collect(),
            products: (0..n).map(|i| (format!("w{i}"), last.clone())).collect(),
        }
    }

    /// P3 drops every pair but the last, each from the live lists in O(1) (round 4, N-1: with
    /// the dead entries read, 61.1 times). Control: without P2, P3 and P4 (P4 alone also drops
    /// them), the n products of n terms reach stage 3, refused.
    #[test]
    fn a_sum_beside_rows_of_one_variable_scale() {
        scale("stale", stale, 250, pass::P2 | pass::P3 | pass::P4, |k| {
            k[..k.len() - 1].iter().all(|&x| !x) && k[k.len() - 1]
        });
    }

    /// The work ([`WORK`]) of [`super::super::products`] on `x - 1 >= 0`, `y >= 0`, the k
    /// rows `z + u{j} >= 0`, on no pair, and after them the k rows `y + (j + 1) z >= 0`, each
    /// paired with `x - 1` for the goal's `x * y`; with the input of [`kept`] and the products
    /// made. The rows on no pair come first, so a search of the rows holding `z` that walks
    /// them meets all k before the first row on a pair (`Formula::and` keeps the order of its
    /// parts); interleaved, it met one.
    fn reads_through_products(k: usize) -> (u64, u64, Vec<super::super::formula::Formula>) {
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
        WORK.with(|n| n.set(0));
        let (made, _) = products(&p, &g, Tier::Cheap).expect("answered");
        (WORK.with(Cell::get), INPUT.with(Cell::get), made)
    }

    /// XR-presburger4 round 2, L1 (Codex), through `products`: the rows on no pair among the
    /// rows holding `z` are not walked, so the work at k and 8k grows by at most 9 times and
    /// stays under 20 a unit of the input of [`kept`]. With every row kept among the holders,
    /// the test of each pair `{x - 1, y + (j + 1) z}` of round 2 entered the k empty lists of
    /// the rows `z + u{j}`: Θ(k²) (63,002 and 4,004,002 reads, VM hm1, job 38).
    #[test]
    fn a_premise_of_rows_on_no_pair_reads_few_lists() {
        let ((small, input, made), large) =
            (reads_through_products(250), reads_through_products(2000).0);
        assert!(!made.is_empty(), "no product made at k = 250");
        assert!(small > 0, "no work counted at k = 250");
        assert!(small <= 20 * input, "{small} units on input {input}");
        assert!(
            9 * small >= large,
            "{small} and {large} units at k = 250 and 2000"
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
    /// `k`, and the kept products hold Θ(m³) = Θ(k^{3/2}) terms. The work of [`kept`]
    /// ([`WORK`]), the products made and their terms, or the refusal; whether [`kept`]
    /// answered, and its input.
    #[allow(clippy::type_complexity)]
    fn grid_through_products(
        m: usize,
        fresh: bool,
    ) -> (u64, Result<(usize, usize), Refusal>, Option<bool>, u64) {
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
        WORK.with(|n| n.set(0));
        INPUT.with(|n| n.set(0));
        ANSWERED.with(|n| n.set(None));
        let made = products(&p, &g, Tier::Cheap).map(|(made, _)| made);
        fn terms(f: &Formula) -> usize {
            match f {
                Formula::Con(c) => c.lin.terms.len(),
                Formula::Not(g) => terms(g),
                Formula::And(fs) | Formula::Or(fs) => fs.iter().map(terms).sum(),
            }
        }
        let made = made.map(|made| (made.len(), made.iter().map(terms).sum()));
        (
            WORK.with(Cell::get),
            made,
            ANSWERED.with(Cell::get),
            INPUT.with(Cell::get),
        )
    }

    /// XR-presburger4 round 2, L6 (Gemini), and META4 orders 3 and 4: the grid family
    /// ([`grid_through_products`]). As Gemini wrote it, `products` cannot build it: its
    /// parallel rows are one row each by then, and no product is made. Its variant with no two
    /// rows parallel is built, and the meter of 40 steps a unit refused it at every size (the
    /// steps were 40.0 times the input, 266,422 and 2,159,522 at m = 45 and 132). Now [`kept`]
    /// answers it (C1 and C2 certify the bicliques, C4 the pairs inside one, P2 drops the
    /// rest), in work that grows by at most 9 times at about 8 times its input, the counts
    /// held exactly (XR-presburger4 round 3, Claude: a ratio alone let a method reading
    /// Θ(products²) pass): work 46,986 and 378,978 (8.07 times) on input 6,660 and 53,988
    /// (8.11 times) at m = 45 and 132 (job j13 of b22-presburger6, with P4). The products kept hold
    /// Θ(k^{3/2}) terms, past 40 times the input of `products` at both sizes, so `products`
    /// still refuses there, by its own check of the terms after [`kept`] has answered
    /// (pending AUTHOR-products-terms-refusal), not by [`kept`]. Negative control:
    /// with C1, C2, C4 and C5 off, [`kept`] refuses at m = 132 (C5 alone certifies the
    /// bicliques too, as their neighbor sum is within its condition).
    #[test]
    fn the_grid_family_is_decided_in_work_linear_in_its_input() {
        let mut counts = Vec::new();
        for m in [45, 132] {
            let (_, made, _, _) = grid_through_products(m, false);
            assert_eq!(
                made,
                Ok((0, 0)),
                "m = {m}: products made from parallel rows"
            );
            let (work, made, answered, input) = grid_through_products(m, true);
            assert_eq!(answered, Some(true), "m = {m}: kept refused");
            eprintln!("grid m = {m}: work {work}, input {input}, made {made:?}");
            counts.push((work, input, made.is_ok()));
        }
        eprintln!("grid counts: {counts:?}");
        assert!(9 * counts[0].0 >= counts[1].0, "{counts:?}");
        assert!(8 * counts[0].1 <= counts[1].1, "{counts:?}");
        assert_eq!(counts, [(46_986, 6_660, false), (378_978, 53_988, false)]);
        OFF.with(|o| o.set(pass::C1 | pass::C2 | pass::C4 | pass::C5));
        let (_, _, answered, _) = grid_through_products(132, true);
        OFF.with(|o| o.set(0));
        assert_eq!(answered, Some(false), "negative control: answered");
    }

    /// The refusal of [`kept`] reaches the caller of [`super::super::entails`] from each tier
    /// of [`super::super::products`]: `x >= 7`, which the goal `x * y >= 7 y` needs with
    /// `y >= 0`, follows from `x + b >= 10`, `x - b - c >= 4` and `c >= 0` together, where `b`
    /// and `c` are factors (of `b * c >= -1000000`), so it is no cheap bound and the goal is
    /// proved at `Tier::Local`. A refusal at `Tier::Cheap` was returned before; one at
    /// `Tier::Local` or `Tier::Whole` was taken for `NotProved` (XR-presburger4 META3,
    /// coordinator point 3), which says the linear reading fails with every product allowed,
    /// never checked. Red with that: `NotProved` at the second call. The tier of each refused
    /// call is checked ([`super::REFUSED_AT`], set from [`super::TIER`]), so the third case is
    /// shown to reach `Tier::Whole` (presburger5's claim most likely wrong, checked).
    #[test]
    fn entailment_refused_for_its_products_says_so() {
        use super::super::formula::{Con, Formula};
        use super::super::{Entailment, Relation, Tier, entails};
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
            super::REFUSED_AT.with(|c| c.set(None));
            super::REFUSE_AFTER.with(|c| c.set(Some(calls)));
            let got = entails(p, g);
            super::REFUSE_AFTER.with(|c| c.set(None));
            (got, super::REFUSED_AT.with(Cell::get))
        };
        for (calls, tier) in [(0, Tier::Cheap), (1, Tier::Local)] {
            let got = refused_after(&p, &g, calls);
            assert_eq!(
                got,
                (Entailment::Refused(Refusal::Products), Some(tier)),
                "{tier:?}"
            );
        }
        // `Tier::Whole` (XR-presburger4 META4 order 4, NB-L1): `x >= 7` follows from
        // `x + 2 z >= 21` and the difference rows `z <= w <= x` together, as `3 x >= 21`. `z`
        // and `w` are factors (of `w * z >= -1000000`), so `part_bounds` keeps them: with
        // neither a factor, its elimination ([`super::super::shadow`]) wrote the three rows
        // as `3 x >= 21` and `Tier::Cheap` proved the goal (job j2 of b22-presburger6: proved
        // with the third call refused). The cluster of `x` holds the general row alone, as `w`
        // is outside it, so neither its rows nor the bounds give `x` a lower side, and only
        // the hull over the whole part does. With the third call of `kept` refused the answer
        // is the refusal; with the fourth, none is made, so the third is the last, at
        // `Tier::Whole`, and the goal is proved there.
        let wz = Lin::atom(Atom::product(&"w".into(), &"z".into()));
        let premise = vec![
            ge0(row(-21, &[(1, "x"), (2, "z")])),
            ge0(row(0, &[(1, "w"), (-1, "z")])),
            ge0(row(0, &[(1, "x"), (-1, "w")])),
            ge0(row(0, &[(1, "y")])),
            ge0(wz.plus(&Lin::constant(1_000_000))),
        ];
        let p = Relation::of(Formula::and(premise));
        assert!(matches!(entails(&p, &g), Entailment::Proved(_)));
        assert!(matches!(refused_after(&p, &g, 3).0, Entailment::Proved(_)));
        let got = refused_after(&p, &g, 2);
        assert_eq!(
            got,
            (Entailment::Refused(Refusal::Products), Some(Tier::Whole)),
            "Whole"
        );
    }

    /// Gemini's family (XR-presburger4 round 3) through [`super::super::products`]: the row
    /// `x + z + w0 + ... + w{k-1} >= 0`, the k rows `x + (j + 1) y >= 0` and the goal
    /// `x * z >= 0`, so the pairs are the sum with itself and with each row `x + (j + 1) y`.
    /// The work of [`kept`], whether it answered, and the products made or the refusal.
    fn sum_holding_the_goal(k: usize) -> (u64, Option<bool>, Result<usize, Refusal>) {
        sum_holding_the_goal_and(k, false)
    }

    /// [`sum_holding_the_goal`], with `y` in the sum too when `with_y` (Gemini's S1_plus_y,
    /// XR-presburger4 round 4, B-L2).
    fn sum_holding_the_goal_and(
        k: usize,
        with_y: bool,
    ) -> (u64, Option<bool>, Result<usize, Refusal>) {
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
        WORK.with(|n| n.set(0));
        ANSWERED.with(|n| n.set(None));
        let made = products(&p, &g, Tier::Cheap).map(|(made, _)| made.len());
        (WORK.with(Cell::get), ANSWERED.with(Cell::get), made)
    }

    /// XR-presburger4 round 3 (Gemini): on [`sum_holding_the_goal`] C2a certifies the k pairs
    /// of the sum with the rows `x + (j + 1) y` (one support, all of one sign) and P2 drops the
    /// sum's pair with itself (each `w{j}` is the sum's own). The work grows by at most 9
    /// times at 8 times `k`, held exactly: 2,347 and 18,447 at k = 100 and 800 (7.86 times;
    /// job j13 of b22-presburger6, with P4). Round 3's method read 10,706 and 645,606 pairs
    /// (60.3 times). The k products kept hold Θ(k²) terms (3 (k + 3) each), past 40 times the
    /// input, so [`super::super::products`] refuses them by its terms check (pending
    /// AUTHOR-products-terms-refusal) after [`kept`] has answered. Negative control: with C2
    /// and C5 off (C5 alone certifies the k pairs too), [`kept`] itself refuses at k = 800
    /// (the k pairs reach stage 3).
    #[test]
    fn a_sum_holding_the_goal_reads_few_pairs() {
        let (small, large) = (sum_holding_the_goal(100), sum_holding_the_goal(800));
        let refused = Err(Refusal::Products);
        eprintln!("sum_holding_the_goal: {small:?} {large:?}");
        assert_eq!(
            (small.clone(), large.clone()),
            (
                (2_347, Some(true), refused.clone()),
                (18_447, Some(true), refused)
            )
        );
        assert!(large.0 <= 9 * small.0, "work {} and {}", small.0, large.0);
        OFF.with(|o| o.set(pass::C2 | pass::C5));
        let control = sum_holding_the_goal(800);
        OFF.with(|o| o.set(0));
        assert_eq!(control.1, Some(false), "negative control: {control:?}");
    }

    /// B-L2 (XR-presburger4 META4 order 2): with `y` in the sum, the sum shares `x` and `y`
    /// with each row `x + (j + 1) y`; the rows still have one support and one sign, so C2a
    /// certifies them and the work grows by at most 9 times at 8 times `k`. Round 4's method
    /// before B-L2 read 62.80 times. Negative control: with C2 and C5 off, [`kept`] refuses.
    #[test]
    fn a_sum_holding_the_goal_and_y_reads_few_pairs() {
        let (small, large) = (
            sum_holding_the_goal_and(100, true),
            sum_holding_the_goal_and(800, true),
        );
        eprintln!("sum_holding_the_goal_and_y: {small:?} {large:?}");
        assert_eq!((small.1, large.1), (Some(true), Some(true)));
        assert!(large.0 <= 9 * small.0, "work {} and {}", small.0, large.0);
        OFF.with(|o| o.set(pass::C2 | pass::C5));
        let control = sum_holding_the_goal_and(800, true);
        OFF.with(|o| o.set(0));
        assert_eq!(control.1, Some(false), "negative control: {control:?}");
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
