//! The region over Q of a conjunction of linear rows over at most two symbols, kept as the
//! rows that bound it (LOAD-chain-move, design A of the cross-review
//! XR-LOAD-chain-move-call-disj-defined-with, META1 A1).
//!
//! Exactness: a row implied over Q by the others has the same rational points on both sides of
//! its removal, so the same integer points too; deciding the kept rows over Z decides the whole
//! conjunction. Dropping rows by any weaker test, as "the rows over the atoms it reads", is not
//! exact (META1, error 1: `z = x && z >= 1` loses `x >= 1`), so a conjunction that reads an
//! `Aux`, a product, a third atom, a disjunction or a negation has no region here and is decided
//! whole by the caller.
//!
//! The region is held as a convex polygon cut from a square of half side `M`, `M` a symbol
//! larger than every number the rows give: a vertex is the meeting point of two sides, its
//! coordinates `(p + q M) / d`, and a sign is read from `q` first and from `p` where `q` is 0.
//! A row is a side of the polygon for every large `M` exactly when it bounds the region over Q
//! along a segment of positive length (a facet), and a full-dimensional region is the meet of
//! its facets, so the rows kept are exactly the facets. The square's own sides are never rows.
//!
//! A region with no interior (a row `==`, two opposite rows meeting on a line, or no point) is
//! not kept: the caller decides those whole. Each region is never changed once made, so a
//! conjunction extended by two callers keeps its region for both (META1, error 2: no cost is
//! amortized over deletions).
//!
//! Complexity tradeoff: each row added reads every side once, O(k) for a region of k sides,
//! and copies the k sides into the new region, O(k) again; no amortization, so a region shared
//! by many extensions costs each the same. A chain whose rows all stay sides (the tangent chain
//! `i x + y >= i^2`) keeps every row, k = n at link n, so Θ(n) a link and Θ(n²) for the chain,
//! with deciding the kept rows on top. The measured moving chain keeps at most 3 rows a link.
//! Prior work: the clip is the one step of Sutherland and Hodgman's polygon clipper (I. E.
//! Sutherland and G. W. Hodgman, "Reentrant polygon clipping", Communications of the ACM
//! 17(1), 1974, doi:10.1145/360767.360802), against one half-plane; a persistent integer
//! hull, worst-case O(log n + log A) a row in any order, is Harvey's (W. Harvey, "Computing
//! two-dimensional integer hulls", SIAM J. Comput. 28(6), 1999, doi:10.1137/S009753979528977X),
//! design B of the cross-review, kept for when a tangent chain must scale.

use std::sync::Arc;

use super::super::formula::Rel;
use super::super::{Atom, Con, Formula, Id};
use zedc_int::int::Int;

/// A side `a x + b y + c0 + c1 M >= 0` of a polygon, with the row it is, `None` for the sides
/// of the square.
#[derive(Clone, Debug)]
struct Side {
    a: Int,
    b: Int,
    c0: Int,
    c1: Int,
    row: Option<Con>,
}

/// The sign of `p + q M` for every large `M`.
fn sign(p: &Int, q: &Int) -> i8 {
    let s = |v: &Int| match () {
        _ if v.is_positive() => 1,
        _ if v.is_negative() => -1,
        _ => 0,
    };
    match s(q) {
        0 => s(p),
        t => t,
    }
}

/// The region over Q of a conjunction of rows over at most two symbols, `axes`, as the sides
/// of its polygon counterclockwise; full-dimensional, every side of positive length.
#[derive(Clone, Debug)]
pub(crate) struct Plane {
    axes: [Option<Atom>; 2],
    sides: Arc<[Side]>,
}

impl Plane {
    /// The square of half side `M`: every point.
    fn all() -> Plane {
        let (z, one, minus) = (Int::from(0), Int::from(1), Int::from(-1));
        let side = |a: &Int, b: &Int| Side {
            a: a.clone(),
            b: b.clone(),
            c0: z.clone(),
            c1: one.clone(),
            row: None,
        };
        Plane {
            axes: [None, None],
            // y >= -M, x <= M, y <= M, x >= -M: counterclockwise.
            sides: Arc::from(vec![
                side(&z, &one),
                side(&minus, &z),
                side(&z, &minus),
                side(&one, &z),
            ]),
        }
    }

    /// The region of `f`, a conjunction of rows `>=` over at most two symbols, when it is
    /// full-dimensional; `None` for any other formula. O(its distinct parts times the sides).
    pub(crate) fn of(f: &Formula) -> Option<Plane> {
        let mut plane = Plane::all();
        let mut sides = plane.sides.to_vec();
        let mut seen = std::collections::HashSet::new();
        let mut stack = vec![f];
        while let Some(g) = stack.pop() {
            match g {
                Formula::Con(c) => plane.add(&mut sides, c)?,
                Formula::And(_) if seen.insert(Id::any(g)) => stack.extend(g.kids()),
                Formula::And(_) => {}
                _ => return None,
            }
        }
        plane.sides = Arc::from(sides);
        Some(plane)
    }

    /// This region with the rows `cons` added, a new region, this one unchanged; `None` when one
    /// is not a row `>=` over the two axes or the region left has no interior. O(k) a row.
    pub(crate) fn with<'a>(&self, cons: impl IntoIterator<Item = &'a Con>) -> Option<Plane> {
        let mut plane = Plane {
            axes: self.axes.clone(),
            sides: self.sides.clone(),
        };
        let mut sides = self.sides.to_vec();
        for c in cons {
            plane.add(&mut sides, c)?;
        }
        plane.sides = Arc::from(sides);
        Some(plane)
    }

    /// The rows kept: the facets of the region.
    pub(crate) fn rows(&self) -> impl Iterator<Item = &Con> {
        self.sides.iter().filter_map(|s| s.row.as_ref())
    }

    /// The axes named so far.
    pub(crate) fn axes(&self) -> impl Iterator<Item = &Atom> {
        self.axes.iter().flatten()
    }

    /// Cuts `sides` by the row `c`, naming its atoms as axes: `None` when `c` is not a row `>=`
    /// over symbols, names a third, or leaves no interior.
    fn add(&mut self, sides: &mut Vec<Side>, c: &Con) -> Option<()> {
        if c.rel != Rel::Ge {
            return None;
        }
        let (mut a, mut b) = (Int::from(0), Int::from(0));
        for (atom, k) in &c.lin.terms {
            if !matches!(atom, Atom::Sym(_)) {
                return None;
            }
            let i = match &self.axes {
                [Some(x), _] if x == atom => 0,
                [_, Some(y)] if y == atom => 1,
                [None, _] => 0,
                [Some(_), None] => 1,
                [Some(_), Some(_)] => return None,
            };
            self.axes[i] = Some(atom.clone());
            *[&mut a, &mut b][i] = k.clone();
        }
        let h = Side {
            a,
            b,
            c0: c.lin.constant.clone(),
            c1: Int::from(0),
            row: Some(c.clone()),
        };
        cut(sides, h)
    }
}

/// The sign of side `h` at the vertex where `e` meets `f`, `e` and `f` not parallel: by
/// Cramer's rule the vertex is `(X / d, Y / d)`, `X = b_e C_f - b_f C_e`, `Y = C_e a_f - C_f a_e`,
/// `d = a_e b_f - a_f b_e`, each `C = c0 + c1 M`, so `h` there is `(a_h X + b_h Y + C_h d) / d`.
fn at(e: &Side, f: &Side, h: &Side) -> i8 {
    let d = &(&e.a * &f.b) - &(&f.a * &e.b);
    let x0 = &(&e.b * &f.c0) - &(&f.b * &e.c0);
    let x1 = &(&e.b * &f.c1) - &(&f.b * &e.c1);
    let y0 = &(&e.c0 * &f.a) - &(&f.c0 * &e.a);
    let y1 = &(&e.c1 * &f.a) - &(&f.c1 * &e.a);
    let n0 = &(&(&h.a * &x0) + &(&h.b * &y0)) + &(&h.c0 * &d);
    let n1 = &(&(&h.a * &x1) + &(&h.b * &y1)) + &(&h.c1 * &d);
    sign(&n0, &n1) * sign(&d, &Int::from(0))
}

/// Cuts the polygon `sides` by the half-plane `h` (Sutherland and Hodgman's step): unchanged
/// when every vertex is on `h`'s side, `h` implied; `None` when none is strictly inside it, so
/// the region left has no interior; otherwise the vertices strictly outside form one run
/// `v_p..v_q` of the cycle, those on the line at its ends with them, and the sides between
/// them are dropped for `h`: `e_p`, `h`, `e_{q+1}`. Vertex `v_i` is where `e_i` meets `e_{i+1}`.
/// The polygon is convex and has positive area, so consecutive sides are never parallel, and
/// `e_p` and `e_{q+1}` each have a vertex strictly inside `h`, so neither is parallel to it.
fn cut(sides: &mut Vec<Side>, h: Side) -> Option<()> {
    let k = sides.len();
    let s: Vec<i8> = (0..k)
        .map(|i| at(&sides[i], &sides[(i + 1) % k], &h))
        .collect();
    if s.iter().all(|&v| v >= 0) {
        return Some(());
    }
    if s.iter().all(|&v| v <= 0) {
        return None;
    }
    // p: the first vertex of the run, its predecessor strictly inside.
    let p = (0..k).find(|&i| s[i] <= 0 && s[(i + k - 1) % k] > 0)?;
    let mut q = p;
    while s[(q + 1) % k] <= 0 {
        q = (q + 1) % k;
    }
    // The new cycle: e_{q+1} .. e_p, then h.
    let mut kept = Vec::with_capacity(k + 1);
    let mut i = (q + 1) % k;
    loop {
        kept.push(sides[i].clone());
        if i == p {
            break;
        }
        i = (i + 1) % k;
    }
    kept.push(h);
    *sides = kept;
    Some(())
}

/// `f` as the formula of its kept rows: the conjunction it is over Q, so over Z too.
pub(crate) fn formula(plane: &Plane) -> Formula {
    Formula::and(plane.rows().map(|c| Formula::Con(c.clone())).collect())
}
