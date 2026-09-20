"""Guillotine-style (edge-to-edge) cutting layout engine.

Model
-----
* Every cut is a straight line spanning the full width or height of the
  sub-rectangle being cut (guillotine / 铡刀 rule).
* Every cut consumes ``kerf`` units of material (the blade/切缝 width).
  The kerf strip sits between the two children of a cut node.
* Pieces may be rotated 90 degrees.
* If the leftover after a cut would be narrower than the kerf it cannot
  yield any usable piece (the blade would eat it entirely), so it is
  treated as trim loss and not represented by a node.

The layout is built with the classic free-rectangle guillotine heuristic:
pieces are sorted largest-first and each piece is placed into the free
rectangle with the best short-side fit, splitting that rectangle with
guillotine cuts.  Because free rectangles are only ever produced by
splitting, the result is always expressible as a binary cut tree.

Locked placements
------------------
Requests may pin some pieces to fixed coordinates.  The locked layout on
the first sheet must itself be guillotine-separable under the same kerf
rules: recursively, a region is either empty, a single piece, or a full
span kerf-wide empty strip separates it into two non-empty groups.
LockedLayoutError is raised when no such decomposition exists; the error
names the pieces of the smallest region that could not be separated.

Cut limit
---------
``max_cuts_per_sheet`` caps the number of cut nodes in any sheet's tree.
The locked layout on the first sheet counts towards the cap and is
rejected when its tree alone exceeds it.  While packing, a free rectangle
is only used when the split it takes keeps its sheet within the cap, and
equal fits prefer the split adding fewer cuts.  A piece that cannot be
cut from a blank sheet within the cap is rejected up front.

Gripper zone
------------
``gripper_zone=(edge, size)`` reserves a full-length strip along one
stock edge (the press gripper/叼口 margin) where no product may be
placed.  Every sheet's tree opens with a full-span cut separating the
strip from the rest of the sheet; the strip becomes a waste leaf, the cut
consumes ``kerf`` and counts towards ``cut_count`` and
``max_cuts_per_sheet``.  All packing and locked-layout decomposition
happen inside the remaining usable region.  Locked pieces entering the
strip or crossing the separating cut are rejected, as is a request whose
usable region is too small for the items.
"""
from __future__ import annotations

from dataclasses import dataclass, field

EPS = 1e-9

# Bound on the number of decomposition attempts while verifying a locked
# layout.  Decomposition can branch exponentially for pathological
# (non-separable) layouts, so give up deterministically past this many
# nodes instead of stalling the service.
LOCKED_SEARCH_BUDGET = 200_000


class UnfitItemError(ValueError):
    """Raised when an item cannot fit on a fresh stock sheet in any orientation."""

    def __init__(self, item_id: str, width: float, height: float,
                 stock_width: float, stock_height: float) -> None:
        self.item_id = item_id
        super().__init__(
            f"item '{item_id}' ({width} x {height}) does not fit on stock "
            f"{stock_width} x {stock_height} in either orientation"
        )


class LockedLayoutError(ValueError):
    """Raised when pinned placements do not form a kerf-valid guillotine layout."""

    def __init__(self, message: str,
                 stuck_group: list[LockedPiece] | None = None) -> None:
        super().__init__(message)
        self.stuck_group = stuck_group


class CutLimitError(ValueError):
    """Raised when the per-sheet cut limit cannot be honoured."""


class GripperZoneError(ValueError):
    """Raised when the gripper zone leaves too little room to pack."""


@dataclass
class Piece:
    """One physical piece to place (an item expanded by its quantity)."""

    item_id: str
    width: float
    height: float


@dataclass
class LockedPiece:
    """A piece pinned to fixed coordinates on the first sheet."""

    piece: Piece
    x: float
    y: float
    width: float
    height: float
    rotated: bool
    ref: int  # 1-based index in the request's locked_placements list


@dataclass
class Node:
    """A node of the binary cut tree.

    kind: "free" (usable offcut, during packing), "waste" (final offcut),
    "item" (a placed piece) or "cut" (internal node with two children).
    """

    kind: str
    x: float
    y: float
    width: float
    height: float
    direction: str | None = None  # cut nodes: "vertical" | "horizontal"
    children: list["Node"] | None = None
    piece: Piece | None = None    # item nodes
    rotated: bool = False         # item nodes
    locked_ref: int | None = None  # item nodes: request index when pinned


@dataclass
class Sheet:
    width: float
    height: float
    kerf: float
    root: Node
    free: list[Node] = field(default_factory=list)
    placements: list[dict] = field(default_factory=list)
    used_area: float = 0.0
    cuts: int = 0  # cut nodes currently in the tree (one per guillotine cut)


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------

def _fits(pw: float, ph: float, width: float, height: float) -> bool:
    return pw <= width + EPS and ph <= height + EPS


def _usable_region(stock_width: float, stock_height: float, kerf: float,
                   gripper: tuple[str, float] | None):
    """Usable rectangle (x, y, width, height) after the gripper zone.

    The reserved strip and the kerf-wide strip its separating cut
    consumes are removed from the sheet; without a gripper zone the whole
    sheet is usable.
    """
    if gripper is None:
        return 0.0, 0.0, stock_width, stock_height
    edge, size = gripper
    size, kerf = float(size), float(kerf)  # keep derived coords floats
    if edge == "top":
        return 0.0, size + kerf, stock_width, stock_height - size - kerf
    if edge == "bottom":
        return 0.0, 0.0, stock_width, stock_height - size - kerf
    if edge == "left":
        return size + kerf, 0.0, stock_width - size - kerf, stock_height
    # right
    return 0.0, 0.0, stock_width - size - kerf, stock_height


def _zone_rect(stock_width: float, stock_height: float,
               gripper: tuple[str, float]):
    """Rectangle (x, y, width, height) reserved for the gripper."""
    edge, size = gripper
    size = float(size)
    if edge == "top":
        return 0.0, 0.0, stock_width, size
    if edge == "bottom":
        return 0.0, stock_height - size, stock_width, size
    if edge == "left":
        return 0.0, 0.0, size, stock_height
    # right
    return stock_width - size, 0.0, size, stock_height


def _contains(a: Node, b: Node) -> bool:
    """True if rectangle a fully contains rectangle b."""
    return (
        a.x <= b.x + EPS
        and a.y <= b.y + EPS
        and a.x + a.width >= b.x + b.width - EPS
        and a.y + a.height >= b.y + b.height - EPS
    )


def _orientations(piece: Piece):
    """Yield (width, height) pairs, unrotated first."""
    yield piece.width, piece.height
    if piece.width != piece.height:
        yield piece.height, piece.width


def _split_cuts(pw: float, ph: float, width: float, height: float,
                kerf: float) -> int:
    """Number of cut nodes ``_fill`` adds for a (pw x ph) corner placement.

    Every usable offcut (wider than the kerf after the cut) costs exactly
    one guillotine cut; remainders not wider than the kerf are trim loss
    and need no cut.
    """
    has_right = width - pw - kerf > EPS
    has_bottom = height - ph - kerf > EPS
    return int(has_right) + int(has_bottom)


def _count_cuts(node: Node) -> int:
    """Number of cut nodes in the subtree rooted at ``node``."""
    if node.kind != "cut":
        return 0
    return 1 + sum(_count_cuts(child) for child in node.children)


def _corner_fill_ok(pw: float, ph: float, width: float, height: float,
                    kerf: float) -> bool:
    """Whether a piece at the leaf's origin yields a kerf-valid tree.

    Two regimes work: the piece reaches the leaf's far edge (slack ~ 0,
    item-only leaf), or at least a kerf plus a non-negative offcut remains
    (cut with a sibling leaf).  A positive sliver narrower than the kerf is
    impossible: the tree derives the cut position from the first child's
    reported width, so trim cannot sit between the piece edge and the kerf.
    """
    slack_r = width - pw
    slack_b = height - ph
    ok_r = slack_r <= EPS or slack_r >= kerf - EPS
    ok_b = slack_b <= EPS or slack_b >= kerf - EPS
    return ok_r and ok_b


# ---------------------------------------------------------------------------
# tree building
# ---------------------------------------------------------------------------

def _become_cut(node: Node, direction: str, children: list[Node]) -> None:
    node.kind = "cut"
    node.direction = direction
    node.children = children
    node.piece = None
    node.rotated = False
    node.locked_ref = None


def _become_item(node: Node, item: Node) -> None:
    node.kind = "item"
    node.width = item.width
    node.height = item.height
    node.piece = item.piece
    node.rotated = item.rotated
    node.locked_ref = item.locked_ref
    node.direction = None
    node.children = None


def _item_node(x: float, y: float, pw: float, ph: float,
               piece: Piece, rotated: bool,
               locked_ref: int | None = None) -> Node:
    return Node(kind="item", x=x, y=y, width=pw, height=ph, piece=piece,
                rotated=rotated, locked_ref=locked_ref)


def _gripper_root(stock_width: float, stock_height: float, kerf: float,
                  gripper: tuple[str, float]) -> tuple[Node, Node]:
    """Root cut node separating the gripper zone from the usable region.

    The separating cut spans the whole sheet and consumes ``kerf``; the
    reserved strip becomes a waste leaf.  For "top"/"left" zones the
    waste leaf is the first child, for "bottom"/"right" the second, so
    the second child always starts exactly ``kerf`` after the first ends.
    Returns (root, usable free node).
    """
    ux, uy, uw, uh = _usable_region(stock_width, stock_height, kerf, gripper)
    zone = Node("waste", *_zone_rect(stock_width, stock_height, gripper))
    usable = Node("free", ux, uy, uw, uh)
    edge = gripper[0]
    direction = "horizontal" if edge in ("top", "bottom") else "vertical"
    children = [zone, usable] if edge in ("top", "left") else [usable, zone]
    root = Node("cut", 0.0, 0.0, stock_width, stock_height,
                direction=direction, children=children)
    return root, usable


def _new_sheet(stock_width: float, stock_height: float, kerf: float,
               gripper: tuple[str, float] | None) -> Sheet:
    """Open a blank sheet, carving out the gripper zone when configured.

    With a gripper zone the root is the separating cut (counted as one
    cut) and the usable region is its free child; otherwise the root
    itself is the single free rectangle.
    """
    if gripper is None:
        root = Node(kind="free", x=0.0, y=0.0,
                    width=stock_width, height=stock_height)
        return Sheet(width=stock_width, height=stock_height, kerf=kerf,
                     root=root, free=[root])
    root, usable = _gripper_root(stock_width, stock_height, kerf, gripper)
    sheet = Sheet(width=stock_width, height=stock_height, kerf=kerf,
                  root=root, free=[usable])
    sheet.cuts = 1  # the gripper zone separating cut
    return sheet


def _fill(node: Node, piece: Piece, pw: float, ph: float, kerf: float,
          rotated: bool | None = None, locked_ref: int | None = None) -> list[Node]:
    """Place a piece (pw x ph) at the top-left corner of free leaf ``node``.

    Mutates ``node`` into an item leaf or a cut subtree and returns the new
    free rectangles created by the split.  Every split consumes ``kerf``
    material; remainders not wider than the kerf are discarded as trim loss.
    ``rotated`` defaults to being derived from the piece's base dimensions.
    """
    x, y, w, h = node.x, node.y, node.width, node.height
    if rotated is None:
        rotated = not (pw == piece.width and ph == piece.height)
    item = _item_node(x, y, pw, ph, piece, rotated, locked_ref)

    right_w = w - pw - kerf    # usable width right of a vertical cut
    bottom_h = h - ph - kerf   # usable height below a horizontal cut
    has_right = right_w > EPS
    has_bottom = bottom_h > EPS

    if not has_right and not has_bottom:
        # The piece consumes the whole rectangle (slivers <= kerf are trim).
        _become_item(node, item)
        return []

    if has_right and has_bottom:
        # Both offcuts are usable.  Two guillotine split orders are possible;
        # keep the one whose full-span offcut is larger.
        if right_w * h >= w * bottom_h:
            # Vertical cut first: right strip spans the full height, then a
            # horizontal cut separates the piece from the offcut below it.
            right = Node("free", x + pw + kerf, y, right_w, h)
            bottom = Node("free", x, y + ph + kerf, pw, bottom_h)
            strip = Node("cut", x, y, pw, h, direction="horizontal",
                         children=[item, bottom])
            _become_cut(node, "vertical", [strip, right])
            return [right, bottom]
        # Horizontal cut first, mirrored.
        right = Node("free", x + pw + kerf, y, right_w, ph)
        bottom = Node("free", x, y + ph + kerf, w, bottom_h)
        strip = Node("cut", x, y, w, ph, direction="vertical",
                     children=[item, right])
        _become_cut(node, "horizontal", [strip, bottom])
        return [bottom, right]

    if has_right:
        right = Node("free", x + pw + kerf, y, right_w, h)
        _become_cut(node, "vertical", [item, right])
        return [right]

    bottom = Node("free", x, y + ph + kerf, w, bottom_h)
    _become_cut(node, "horizontal", [item, bottom])
    return [bottom]


def _prune(free: list[Node], limit: int = 200) -> None:
    """Drop free rectangles fully contained in another (keeps packing fast).

    This never affects the guillotine property: it only removes redundant
    placement candidates.  Skipped for very large lists to bound the O(n^2)
    cost.
    """
    if len(free) > limit:
        return
    kept = []
    for i, rect in enumerate(free):
        dropped = False
        for j, other in enumerate(free):
            if i == j:
                continue
            # Drop rect if strictly inside another, or a duplicate of an
            # earlier one.
            if _contains(other, rect) and (not _contains(rect, other) or j < i):
                dropped = True
                break
        if not dropped:
            kept.append(rect)
    free[:] = kept


def _best_fit(sheets: list[Sheet], piece: Piece,
              max_cuts: int | None = None):
    """Find the best placement for a piece across all open sheets.

    Returns (sheet_index, free_rect_index, pw, ph) or None.  Scoring is
    best-short-side-fit with best-area-fit as tie-breaker.  When
    ``max_cuts`` is set, rectangles whose split would push their sheet
    past the limit are skipped, and equal fits prefer the split adding
    fewer cuts.
    """
    best = None
    for si, sheet in enumerate(sheets):
        for fi, rect in enumerate(sheet.free):
            for pw, ph in _orientations(piece):
                if not _fits(pw, ph, rect.width, rect.height):
                    continue
                if not _corner_fill_ok(pw, ph, rect.width, rect.height,
                                       sheet.kerf):
                    # Fitting here would leave an unrepresentable trim
                    # sliver between the piece and the kerf; skip it.
                    continue
                short_side = min(rect.width - pw, rect.height - ph)
                area_left = rect.width * rect.height - pw * ph
                if max_cuts is None:
                    score = (short_side, area_left)
                else:
                    added = _split_cuts(pw, ph, rect.width, rect.height,
                                        sheet.kerf)
                    if sheet.cuts + added > max_cuts:
                        # The split would exceed the sheet's cut limit.
                        continue
                    score = (short_side, area_left, added)
                if best is None or score < best[0]:
                    best = (score, si, fi, pw, ph)
    if best is None:
        return None
    _, si, fi, pw, ph = best
    return si, fi, pw, ph


def _place(sheet: Sheet, free_idx: int, piece: Piece, pw: float, ph: float) -> None:
    rect = sheet.free[free_idx]
    sheet.cuts += _split_cuts(pw, ph, rect.width, rect.height, sheet.kerf)
    sheet.free.pop(free_idx)
    new_rects = _fill(rect, piece, pw, ph, sheet.kerf)
    sheet.free.extend(new_rects)
    _prune(sheet.free)
    sheet.used_area += pw * ph
    sheet.placements.append({
        "item": piece.item_id,
        "x": _num(rect.x),
        "y": _num(rect.y),
        "width": _num(pw),
        "height": _num(ph),
        "rotated": not (pw == piece.width and ph == piece.height),
        "locked": False,
    })


def _fresh_orientation(piece: Piece, stock_width: float, stock_height: float,
                       kerf: float, max_cuts: int | None, base_cuts: int = 0):
    """Pick (width, height) for a piece starting a blank sheet.

    Without a cut limit the unrotated orientation is preferred (a fresh
    sheet always fits; sizes are checked up front).  With a limit,
    orientations whose split would exceed it are discarded and the one
    needing the fewest cuts wins, unrotated on ties, leaving cut budget
    for the pieces that follow.  ``base_cuts`` counts cuts a blank sheet
    already carries (the gripper zone separating cut).
    """
    if max_cuts is None:
        pw, ph = piece.width, piece.height
        if not _fits(pw, ph, stock_width, stock_height):
            pw, ph = ph, pw
        return pw, ph
    best = None
    for pw, ph in _orientations(piece):
        if not _fits(pw, ph, stock_width, stock_height):
            continue
        cuts = _split_cuts(pw, ph, stock_width, stock_height, kerf)
        if base_cuts + cuts > max_cuts:
            continue
        if best is None or cuts < best[0]:
            best = (cuts, pw, ph)
    if best is None:
        # Every item is checked against a blank sheet up front, so a
        # fitting orientation within the limit always exists here.
        raise CutLimitError(
            f"item '{piece.item_id}' ({piece.width} x {piece.height}) cannot "
            f"be cut from a fresh sheet {stock_width} x {stock_height} "
            f"within max_cuts_per_sheet {max_cuts}"
        )
    return best[1], best[2]


# ---------------------------------------------------------------------------
# locked layout: guillotine decomposition of pinned pieces
# ---------------------------------------------------------------------------

def _locked_labels(group: list[LockedPiece]) -> list[str]:
    """Distinct request references of a stuck group, input order preserved."""
    labels = []
    for lp in group:
        label = f"#{lp.ref} item '{lp.piece.item_id}'"
        if label not in labels:
            labels.append(label)
    return labels


def _check_locked_gripper_clearance(locked_pieces: list[LockedPiece],
                                    ux: float, uy: float, uw: float,
                                    uh: float, stock_width: float,
                                    stock_height: float,
                                    gripper: tuple[str, float]) -> None:
    """Pinned pieces must lie entirely inside the usable region.

    A piece jutting into the reserved strip enters the gripper zone; one
    that only reaches the kerf-wide strip the separating cut consumes
    crosses that cut.  Both are rejected naming the request reference.
    """
    edge, size = gripper
    zx, zy, zw, zh = _zone_rect(stock_width, stock_height, gripper)
    for lp in locked_pieces:
        if (lp.x >= ux - EPS and lp.y >= uy - EPS
                and lp.x + lp.width <= ux + uw + EPS
                and lp.y + lp.height <= uy + uh + EPS):
            continue
        enters = (lp.x < zx + zw - EPS and lp.x + lp.width > zx + EPS
                  and lp.y < zy + zh - EPS and lp.y + lp.height > zy + EPS)
        reason = (
            "enters the gripper zone" if enters else
            "crosses the gripper zone cut (the kerf-wide strip consumed "
            "by the separating cut)"
        )
        raise LockedLayoutError(
            f"locked placement #{lp.ref} item '{lp.piece.item_id}' at "
            f"({_num(lp.x)}, {_num(lp.y)}) ({_num(lp.width)} x "
            f"{_num(lp.height)}) {reason}: locked pieces must lie "
            f"entirely within the usable area {_num(uw)} x {_num(uh)} at "
            f"({_num(ux)}, {_num(uy)}) left by the gripper zone "
            f"(edge '{edge}', size {_num(size)})",
            stuck_group=[lp],
        )


def _isolate_locked(node: Node, lp: LockedPiece, kerf: float) -> list[Node]:
    """Fill a free region containing exactly one pinned piece.

    Peels free strips off the top/left edges (each cut consumes ``kerf``)
    until the piece is flush with the region's origin, then splits the
    right/bottom offcuts with the usual corner fill.  Returns the new free
    rectangles.

    A gap of exactly 0 needs no cut; a gap larger than ``kerf`` leaves a
    free strip (gap minus kerf).  A gap in between is impossible: the blade
    is wider than the clearance but a sliver that small cannot stay as an
    origin-side leaf, so the pinned layout is not guillotine-separable.
    """
    free_rects: list[Node] = []
    top_gap = lp.y - node.y
    left_gap = lp.x - node.x
    if not (left_gap <= EPS or left_gap > kerf + EPS) \
            or not (top_gap <= EPS or top_gap > kerf + EPS):
        side = "left" if 0 < left_gap <= kerf + EPS else "top"
        raise LockedLayoutError(
            f"locked placement #{lp.ref} item '{lp.piece.item_id}' leaves "
            f"only {_num(left_gap if side == 'left' else top_gap)} units of "
            f"clearance on its {side} side, less than the kerf ({kerf}): no "
            "guillotine cut can separate it without cutting into the piece",
            stuck_group=[lp],
        )

    # Peel order mirrors _fill's heuristic: larger usable strip first.
    # The cut at (origin + gap - kerf) separates a free strip of width
    # (gap - kerf) from the kerf [origin+gap-kerf, origin+gap); the inner
    # region begins at the piece's edge (origin + gap) and keeps the
    # region's far edge, so slack past the piece is handled by _fill.
    if left_gap > EPS and top_gap > EPS \
            and (left_gap - kerf) * node.height < (top_gap - kerf) * node.width:
        # Horizontal (top) peel first.
        strip_w = top_gap - kerf
        inner = Node("free", node.x, node.y + top_gap,
                     node.width, node.height - top_gap)
        strip = Node("free", node.x, node.y, node.width, strip_w)
        _become_cut(node, "horizontal", [strip, inner])
        free_rects.append(strip)
        node = inner
        top_gap = 0.0

    while left_gap > EPS:
        strip_w = left_gap - kerf
        inner = Node("free", node.x + left_gap, node.y,
                     node.width - left_gap, node.height)
        strip = Node("free", node.x, node.y, strip_w, node.height)
        _become_cut(node, "vertical", [strip, inner])
        free_rects.append(strip)
        node = inner
        left_gap = 0.0

    while top_gap > EPS:
        strip_w = top_gap - kerf
        inner = Node("free", node.x, node.y + top_gap,
                     node.width, node.height - top_gap)
        strip = Node("free", node.x, node.y, node.width, strip_w)
        _become_cut(node, "horizontal", [strip, inner])
        free_rects.append(strip)
        node = inner
        top_gap = 0.0

    free_rects.extend(_fill(
        node, lp.piece, lp.width, lp.height, kerf,
        rotated=lp.rotated, locked_ref=lp.ref,
    ))
    return free_rects


def _valid_region(node: Node, group: list[LockedPiece]) -> bool:
    """Every pinned piece of the group must lie inside the region."""
    rx2, ry2 = node.x + node.width, node.y + node.height
    for lp in group:
        if (lp.x < node.x - EPS or lp.y < node.y - EPS
                or lp.x + lp.width > rx2 + EPS
                or lp.y + lp.height > ry2 + EPS):
            return False
    return True


def _decompose_locked(node: Node, group: list[LockedPiece],
                      kerf: float, budget: list[int]) -> list[Node]:
    """Recursively turn free ``node`` into a cut tree over ``group``.

    Returns the usable free leaves created by the decomposition.  Raises
    LockedLayoutError if the pinned layout cannot be produced by full-span
    guillotine cuts separated by ``kerf``-wide strips.
    """
    budget[0] -= 1
    if budget[0] <= 0:
        raise LockedLayoutError(
            "locked placements are too complex to verify: could not derive a "
            "guillotine cut tree within the search budget"
        )
    if not group:
        return [node]
    if len(group) == 1:
        return _isolate_locked(node, group[0], kerf)

    x, y, w, h = node.x, node.y, node.width, node.height

    # Candidate cut positions: a full-span, kerf-wide empty strip must
    # separate the two groups.  Around any piece edge ``e`` the strip's near
    # edge is tried at e - kerf (strip ending at the edge), e (strip starting
    # at the edge) and e + kerf.  These three positions cover every feasible
    # way of allocating the slack on either side of the groups; e + kerf in
    # particular keeps the offcut past a tight group inside the origin-side
    # child as a usable free leaf instead of leaving it as trim.
    candidates = []

    def _add_candidate(direction: str, cut_at: float, groups) -> None:
        """groups is (first, second) already split at the kerf strip."""
        first_group, second_group = groups
        if not first_group or not second_group:
            return
        if direction == "vertical":
            if cut_at < x + EPS or cut_at + kerf > x + w - EPS:
                return
            a = cut_at - x
            if a <= EPS or w - a - kerf <= EPS:
                return
            key = ("v", round(cut_at, 9))
            if any(c[0] == key for c in candidates):
                return
            candidates.append((key, "vertical", cut_at, a,
                               w - a - kerf, first_group, second_group))
        else:
            if cut_at < y + EPS or cut_at + kerf > y + h - EPS:
                return
            a = cut_at - y
            if a <= EPS or h - a - kerf <= EPS:
                return
            key = ("h", round(cut_at, 9))
            if any(c[0] == key for c in candidates):
                return
            candidates.append((key, "horizontal", cut_at, a,
                               h - a - kerf, first_group, second_group))

    def _split_vertical(cut_at: float):
        left, right = [], []
        for lp in group:
            if lp.x + lp.width <= cut_at + EPS:
                left.append(lp)
            elif lp.x >= cut_at + kerf - EPS:
                right.append(lp)
            else:
                return None
        return left, right

    def _split_horizontal(cut_at: float):
        top, bottom = [], []
        for lp in group:
            if lp.y + lp.height <= cut_at + EPS:
                top.append(lp)
            elif lp.y >= cut_at + kerf - EPS:
                bottom.append(lp)
            else:
                return None
        return top, bottom

    x_edges = sorted({round(lp.x, 9) for lp in group}
                     | {round(lp.x + lp.width, 9) for lp in group})
    for edge in x_edges:
        for cut_at in (edge, edge - kerf, edge + kerf):
            split = _split_vertical(cut_at)
            if split is not None:
                _add_candidate("vertical", cut_at, split)

    y_edges = sorted({round(lp.y, 9) for lp in group}
                     | {round(lp.y + lp.height, 9) for lp in group})
    for edge in y_edges:
        for cut_at in (edge, edge - kerf, edge + kerf):
            split = _split_horizontal(cut_at)
            if split is not None:
                _add_candidate("horizontal", cut_at, split)

    # Far-edge cuts first: making the origin-side child large keeps the
    # slack between a tight group and the stock edge inside that child as a
    # usable free leaf, instead of discarding it as trim behind the tight
    # child.  Ties: fewer pieces in the first child, then axis, then edge.
    candidates.sort(key=lambda c: (-(c[3] * (h if c[1] == "vertical" else w)),
                                   len(c[5]), c[1], c[3]))

    smallest_stuck: list[LockedPiece] | None = None

    def _consider(exc: LockedLayoutError) -> None:
        nonlocal smallest_stuck
        stuck = exc.stuck_group if exc.stuck_group is not None else group
        if smallest_stuck is None or len(stuck) < len(smallest_stuck):
            smallest_stuck = stuck

    for _, direction, _edge, a, b, first_group, second_group in candidates:
        if direction == "vertical":
            first = Node("free", x, y, a, h)
            second = Node("free", x + a + kerf, y, b, h)
        else:
            first = Node("free", x, y, w, a)
            second = Node("free", x, y + a + kerf, w, b)
        if not _valid_region(first, first_group) \
                or not _valid_region(second, second_group):
            continue
        try:
            free_first = _decompose_locked(first, first_group, kerf, budget)
            free_second = _decompose_locked(second, second_group, kerf, budget)
        except LockedLayoutError as exc:
            _consider(exc)
            continue
        _become_cut(node, direction, [first, second])
        return free_first + free_second

    stuck_group = smallest_stuck if smallest_stuck is not None else group
    labels = ", ".join(_locked_labels(stuck_group))
    raise LockedLayoutError(
        "locked placements cannot be produced by guillotine cuts with "
        f"kerf {kerf}: no full-span cut (leaving a {kerf}-wide kerf strip) "
        f"separates the conflicting placements ({labels}); adjust their "
        "coordinates or the kerf",
        stuck_group=stuck_group,
    )


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------

def _num(value: float) -> float:
    """Round away floating point dust from coordinate arithmetic."""
    return round(value, 6)


def _finalize(node: Node) -> None:
    """Turn remaining free leaves into waste leaves."""
    if node.kind == "free":
        node.kind = "waste"
        return
    if node.kind == "cut":
        for child in node.children:
            _finalize(child)


def _node_dict(node: Node) -> dict:
    base = {
        "x": _num(node.x),
        "y": _num(node.y),
        "width": _num(node.width),
        "height": _num(node.height),
    }
    if node.kind == "cut":
        return {
            "type": "cut",
            "direction": node.direction,
            **base,
            "children": [_node_dict(child) for child in node.children],
        }
    if node.kind == "item":
        return {
            "type": "item",
            "item": node.piece.item_id,
            "rotated": node.rotated,
            **base,
        }
    return {"type": "waste", **base}


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def build_plan(stock_width: float, stock_height: float, kerf: float,
               items: list[tuple[str, float, float, int]],
               locked: list[tuple[str, float, float, bool, int]] | None = None,
               max_cuts_per_sheet: int | None = None,
               gripper_zone: tuple[str, float] | None = None
               ) -> dict:
    """Pack ``items`` (id, width, height, quantity) onto stock sheets.

    ``locked`` optionally pins pieces on the first sheet; each tuple is
    (item_id, x, y, rotated, request_index).  Locked pieces count towards
    their item's quantity.  ``max_cuts_per_sheet`` caps the number of cuts
    (cut nodes) in any sheet's tree; the locked layout on the first sheet
    counts towards the cap.  ``gripper_zone`` optionally reserves a
    full-length strip along one stock edge as ``(edge, size)`` with edge
    in "top"/"right"/"bottom"/"left": every sheet's tree opens with a
    full-span cut separating the strip as a waste leaf, and the cut
    consumes ``kerf`` and counts towards the cap.  Returns a dict
    matching the PlanResponse schema.  Raises UnfitItemError if any item
    cannot fit on a fresh sheet, GripperZoneError if the reserved strip
    leaves too little room, CutLimitError if the cut cap cannot be
    honoured, or LockedLayoutError if the pinned layout is invalid.
    """
    ux, uy, uw, uh = _usable_region(stock_width, stock_height, kerf,
                                    gripper_zone)
    if gripper_zone is not None and (uw <= EPS or uh <= EPS):
        edge, size = gripper_zone
        raise GripperZoneError(
            f"gripper zone (edge '{edge}', size {size}) with kerf {kerf} "
            f"leaves no usable area on stock {stock_width} x "
            f"{stock_height}: the reserved strip plus the separating cut "
            "consume the whole sheet"
        )

    catalog: dict[str, Piece] = {}
    for item_id, width, height, quantity in items:
        if not (_fits(width, height, uw, uh)
                or _fits(height, width, uw, uh)):
            if gripper_zone is None:
                raise UnfitItemError(item_id, width, height,
                                     stock_width, stock_height)
            edge, size = gripper_zone
            raise GripperZoneError(
                f"item '{item_id}' ({width} x {height}) does not fit in "
                f"the usable area {_num(uw)} x {_num(uh)} left on stock "
                f"{stock_width} x {stock_height} by the gripper zone "
                f"(edge '{edge}', size {size}, kerf {kerf}) in either "
                "orientation"
            )
        if max_cuts_per_sheet is not None:
            piece = Piece(item_id, width, height)
            gripper_cut = 1 if gripper_zone is not None else 0
            min_cuts = gripper_cut + min(
                _split_cuts(pw, ph, uw, uh, kerf)
                for pw, ph in _orientations(piece)
                if _fits(pw, ph, uw, uh)
            )
            if min_cuts > max_cuts_per_sheet:
                if gripper_zone is None:
                    sheet_desc = f"{stock_width} x {stock_height}"
                else:
                    sheet_desc = (
                        f"(usable area {_num(uw)} x {_num(uh)} after the "
                        "gripper zone cut)"
                    )
                raise CutLimitError(
                    f"item '{item_id}' ({width} x {height}) needs at least "
                    f"{min_cuts} cuts on a fresh sheet {sheet_desc}, "
                    f"exceeding max_cuts_per_sheet {max_cuts_per_sheet}"
                )
        catalog[item_id] = Piece(item_id, width, height)

    locked_pieces: list[LockedPiece] = []
    for item_id, x, y, rotated, ref in locked or []:
        x, y = float(x), float(y)
        piece = catalog[item_id]
        pw, ph = (piece.height, piece.width) if rotated else (
            piece.width, piece.height)
        if (x < -EPS or y < -EPS
                or x + pw > stock_width + EPS
                or y + ph > stock_height + EPS):
            raise LockedLayoutError(
                f"locked placement #{ref} item '{item_id}' at ({_num(x)}, "
                f"{_num(y)}) ({_num(pw)} x {_num(ph)}) is outside the stock "
                f"{stock_width} x {stock_height}"
            )
        locked_pieces.append(LockedPiece(
            piece=piece, x=x, y=y, width=pw, height=ph,
            rotated=bool(rotated), ref=ref,
        ))

    if gripper_zone is not None:
        _check_locked_gripper_clearance(locked_pieces, ux, uy, uw, uh,
                                        stock_width, stock_height,
                                        gripper_zone)

    # Pairwise overlaps are rejected explicitly with both references named.
    for i in range(len(locked_pieces)):
        a = locked_pieces[i]
        for j in range(i + 1, len(locked_pieces)):
            b = locked_pieces[j]
            overlap_x = min(a.x + a.width, b.x + b.width) - max(a.x, b.x)
            overlap_y = min(a.y + a.height, b.y + b.height) - max(a.y, b.y)
            if overlap_x > EPS and overlap_y > EPS:
                raise LockedLayoutError(
                    f"locked placements #{a.ref} item '{a.piece.item_id}' and "
                    f"#{b.ref} item '{b.piece.item_id}' overlap"
                )

    # Remaining quantities after the locked ones are consumed.
    remaining: list[Piece] = []
    for item_id, width, height, quantity in items:
        locked_count = sum(1 for lp in locked_pieces
                           if lp.piece.item_id == item_id)
        remaining.extend(Piece(item_id, width, height)
                         for _ in range(quantity - locked_count))

    sheets: list[Sheet] = []
    total_pieces = len(locked_pieces) + len(remaining)

    if locked_pieces:
        sheet = _new_sheet(stock_width, stock_height, kerf, gripper_zone)
        sheets.append(sheet)
        region = sheet.free.pop()  # the decomposition consumes this leaf
        free_rects = _decompose_locked(
            region, locked_pieces, kerf, [LOCKED_SEARCH_BUDGET])
        sheet.cuts = _count_cuts(sheet.root)
        if max_cuts_per_sheet is not None and sheet.cuts > max_cuts_per_sheet:
            note = (" (including the gripper zone cut)"
                    if gripper_zone is not None else "")
            raise CutLimitError(
                f"locked placements need {sheet.cuts} cuts on the first "
                f"sheet{note}, exceeding max_cuts_per_sheet "
                f"{max_cuts_per_sheet}"
            )
        sheet.free.extend(free_rects)
        _prune(sheet.free)
        for lp in sorted(locked_pieces, key=lambda p: (p.y, p.x)):
            sheet.used_area += lp.width * lp.height
            sheet.placements.append({
                "item": lp.piece.item_id,
                "x": _num(lp.x),
                "y": _num(lp.y),
                "width": _num(lp.width),
                "height": _num(lp.height),
                "rotated": lp.rotated,
                "locked": True,
            })

    # Largest first: longer max side, then larger area.
    remaining.sort(key=lambda p: (max(p.width, p.height), p.width * p.height),
                   reverse=True)

    for piece in remaining:
        found = _best_fit(sheets, piece, max_cuts_per_sheet)
        if found is None:
            sheet = _new_sheet(stock_width, stock_height, kerf, gripper_zone)
            sheets.append(sheet)
            pw, ph = _fresh_orientation(piece, uw, uh, kerf,
                                        max_cuts_per_sheet,
                                        base_cuts=sheet.cuts)
            _place(sheet, 0, piece, pw, ph)
        else:
            si, fi, pw, ph = found
            _place(sheets[si], fi, piece, pw, ph)

    sheet_area = stock_width * stock_height
    result_sheets = []
    total_used = 0.0
    for index, sheet in enumerate(sheets):
        _finalize(sheet.root)
        total_used += sheet.used_area
        result_sheets.append({
            "index": index,
            "width": _num(stock_width),
            "height": _num(stock_height),
            "utilization": round(sheet.used_area / sheet_area, 6),
            "cut_count": _count_cuts(sheet.root),
            "placements": sheet.placements,
            "cut_tree": _node_dict(sheet.root),
        })

    return {
        "stock": {"width": _num(stock_width), "height": _num(stock_height)},
        "kerf": _num(kerf),
        "sheets": result_sheets,
        "summary": {
            "sheet_count": len(sheets),
            "piece_count": total_pieces,
            "used_area": _num(total_used),
            "stock_area": _num(len(sheets) * sheet_area),
            "utilization": round(total_used / (len(sheets) * sheet_area), 6),
        },
    }
