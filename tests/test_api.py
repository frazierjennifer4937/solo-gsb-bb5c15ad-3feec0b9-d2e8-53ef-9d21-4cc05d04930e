from fastapi.testclient import TestClient

from app.main import app
from tests.check_plan import check_plan, iter_leaves

client = TestClient(app)

VALID_PAYLOAD = {
    "stock": {"width": 900, "height": 600},
    "kerf": 3,
    "items": [
        {"id": "flyer", "width": 210, "height": 148, "quantity": 8},
        {"id": "card", "width": 90, "height": 55, "quantity": 20},
    ],
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_create_plan_happy_path():
    response = client.post("/plans", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["stock"] == {"width": 900.0, "height": 600.0}
    assert body["kerf"] == 3.0
    assert body["summary"]["piece_count"] == 28
    assert body["summary"]["sheet_count"] == len(body["sheets"])
    check_plan(body)


def test_create_plan_auto_generates_item_ids():
    payload = {
        "stock": {"width": 100, "height": 100},
        "items": [{"width": 50, "height": 50, "quantity": 2}],
    }
    response = client.post("/plans", json=payload)
    assert response.status_code == 200
    items = {p["item"] for s in response.json()["sheets"] for p in s["placements"]}
    assert items == {"item-1"}


def test_unfit_item_returns_clear_error():
    payload = {
        "stock": {"width": 100, "height": 100},
        "kerf": 2,
        "items": [{"id": "poster", "width": 150, "height": 80, "quantity": 1}],
    }
    response = client.post("/plans", json=payload)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "poster" in detail
    assert "150" in detail and "100" in detail


def test_invalid_stock_dimensions_rejected():
    payload = {"stock": {"width": -10, "height": 100},
               "items": [{"width": 10, "height": 10, "quantity": 1}]}
    assert client.post("/plans", json=payload).status_code == 422

    payload["stock"] = {"width": 100, "height": 0}
    assert client.post("/plans", json=payload).status_code == 422


def test_invalid_item_dimensions_rejected():
    payload = {"stock": {"width": 100, "height": 100},
               "items": [{"width": 0, "height": 10, "quantity": 1}]}
    assert client.post("/plans", json=payload).status_code == 422


def test_invalid_quantity_rejected():
    payload = {"stock": {"width": 100, "height": 100},
               "items": [{"width": 10, "height": 10, "quantity": 0}]}
    assert client.post("/plans", json=payload).status_code == 422


def test_negative_kerf_rejected():
    payload = {"stock": {"width": 100, "height": 100}, "kerf": -1,
               "items": [{"width": 10, "height": 10, "quantity": 1}]}
    assert client.post("/plans", json=payload).status_code == 422


def test_empty_item_list_rejected():
    payload = {"stock": {"width": 100, "height": 100}, "items": []}
    assert client.post("/plans", json=payload).status_code == 422


def test_duplicate_item_ids_rejected():
    payload = {
        "stock": {"width": 100, "height": 100},
        "items": [
            {"id": "A", "width": 10, "height": 10, "quantity": 1},
            {"id": "A", "width": 20, "height": 20, "quantity": 1},
        ],
    }
    assert client.post("/plans", json=payload).status_code == 422


def test_missing_fields_rejected():
    assert client.post("/plans", json={}).status_code == 422
    assert client.post("/plans", json={"stock": {"width": 100}}).status_code == 422


# ---------------------------------------------------------------------------
# locked placements
# ---------------------------------------------------------------------------

LOCKED_PAYLOAD = {
    "stock": {"width": 100, "height": 100},
    "kerf": 5,
    "items": [{"id": "A", "width": 45, "height": 40, "quantity": 3}],
    "locked_placements": [
        {"item": "A", "x": 0, "y": 0, "rotated": False},
        {"item": "A", "x": 50, "y": 0, "rotated": False},
    ],
}


def test_locked_placements_happy_path():
    response = client.post("/plans", json=LOCKED_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    check_plan(body)
    sheet0 = body["sheets"][0]
    locked = [p for p in sheet0["placements"] if p["locked"]]
    unlocked = [p for p in sheet0["placements"] if not p["locked"]]
    assert len(locked) == 2
    assert {(p["x"], p["y"]) for p in locked} == {(0.0, 0.0), (50.0, 0.0)}
    # One remaining item of the quantity is still placed somewhere.
    assert body["summary"]["piece_count"] == 3
    assert len(locked) + len(unlocked) <= 3


def test_locked_defaults_rotated_to_false():
    payload = {
        "stock": {"width": 100, "height": 100},
        "items": [{"id": "A", "width": 40, "height": 40, "quantity": 1}],
        "locked_placements": [{"item": "A", "x": 0, "y": 0}],
    }
    body = client.post("/plans", json=payload).json()
    assert body["sheets"][0]["placements"][0]["rotated"] is False


def test_locked_placements_without_field_keep_working():
    response = client.post("/plans", json=VALID_PAYLOAD)
    assert response.status_code == 200
    assert all(p["locked"] is False
               for s in response.json()["sheets"] for p in s["placements"])


def test_locked_reference_to_unknown_item_rejected():
    payload = {
        "stock": {"width": 100, "height": 100},
        "items": [{"id": "A", "width": 10, "height": 10, "quantity": 1}],
        "locked_placements": [{"item": "B", "x": 0, "y": 0}],
    }
    response = client.post("/plans", json=payload)
    assert response.status_code == 422
    assert "B" in response.json()["detail"]


def test_locked_count_exceeding_quantity_rejected():
    payload = {
        "stock": {"width": 100, "height": 100},
        "items": [{"id": "A", "width": 10, "height": 10, "quantity": 1}],
        "locked_placements": [
            {"item": "A", "x": 0, "y": 0},
            {"item": "A", "x": 50, "y": 50},
        ],
    }
    response = client.post("/plans", json=payload)
    assert response.status_code == 422
    assert "#2" in response.json()["detail"]


def test_locked_overlap_rejected():
    payload = {
        "stock": {"width": 100, "height": 100},
        "items": [{"id": "A", "width": 40, "height": 40, "quantity": 2}],
        "locked_placements": [
            {"item": "A", "x": 0, "y": 0},
            {"item": "A", "x": 30, "y": 30},
        ],
    }
    response = client.post("/plans", json=payload)
    assert response.status_code == 422
    assert "#1" in response.json()["detail"]
    assert "#2" in response.json()["detail"]


def test_locked_out_of_bounds_rejected():
    payload = {
        "stock": {"width": 100, "height": 100},
        "items": [{"id": "A", "width": 40, "height": 40, "quantity": 1}],
        "locked_placements": [{"item": "A", "x": 70, "y": 70}],
    }
    assert client.post("/plans", json=payload).status_code == 422


def test_locked_non_guillotine_layout_rejected():
    # Four interlocking blades: non-slicing, even with zero kerf.
    coords = [(0, 0, 80, 120), (80, 0, 120, 80),
              (160, 80, 40, 120), (0, 120, 160, 80)]
    payload = {
        "stock": {"width": 200, "height": 200},
        "kerf": 0,
        "items": [
            {"id": f"p{i}", "width": w, "height": h, "quantity": 1}
            for i, (x, y, w, h) in enumerate(coords, 1)
        ],
        "locked_placements": [
            {"item": f"p{i}", "x": x, "y": y}
            for i, (x, y, w, h) in enumerate(coords, 1)
        ],
    }
    response = client.post("/plans", json=payload)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "p1" in detail and "guillotine" in detail


def test_locked_touching_with_kerf_rejected():
    payload = {
        "stock": {"width": 100, "height": 100},
        "kerf": 3,
        "items": [{"id": "A", "width": 45, "height": 40, "quantity": 2}],
        "locked_placements": [
            {"item": "A", "x": 0, "y": 0},
            {"item": "A", "x": 45, "y": 0},
        ],
    }
    response = client.post("/plans", json=payload)
    assert response.status_code == 422
    assert "#1" in response.json()["detail"]


def test_locked_placements_all_on_first_sheet():
    response = client.post("/plans", json=LOCKED_PAYLOAD)
    body = response.json()
    for index, sheet in enumerate(body["sheets"]):
        for p in sheet["placements"]:
            if p["locked"]:
                assert index == 0


def test_locked_negative_coordinates_rejected():
    payload = {
        "stock": {"width": 100, "height": 100},
        "items": [{"id": "A", "width": 10, "height": 10, "quantity": 1}],
        "locked_placements": [{"item": "A", "x": -1, "y": 0}],
    }
    assert client.post("/plans", json=payload).status_code == 422


# ---------------------------------------------------------------------------
# max_cuts_per_sheet
# ---------------------------------------------------------------------------

def test_max_cuts_per_sheet_limits_cuts():
    payload = {
        "stock": {"width": 100, "height": 100},
        "kerf": 0,
        "items": [{"id": "A", "width": 50, "height": 100, "quantity": 4}],
        "max_cuts_per_sheet": 1,
    }
    response = client.post("/plans", json=payload)
    assert response.status_code == 200
    body = response.json()
    check_plan(body)
    assert body["summary"]["sheet_count"] == 2
    assert all(s["cut_count"] <= 1 for s in body["sheets"])


def test_max_cuts_per_sheet_rejects_uncuttable_item():
    # A 40x40 piece on a blank 100x100 sheet (kerf 3) needs two cuts.
    payload = {
        "stock": {"width": 100, "height": 100},
        "kerf": 3,
        "items": [{"id": "A", "width": 40, "height": 40, "quantity": 1}],
        "max_cuts_per_sheet": 1,
    }
    response = client.post("/plans", json=payload)
    assert response.status_code == 422
    assert "max_cuts_per_sheet" in response.json()["detail"]


def test_max_cuts_per_sheet_counts_locked_tree():
    # The locked pair needs three cuts on the first sheet.
    payload = {
        "stock": {"width": 100, "height": 100},
        "kerf": 5,
        "items": [{"id": "A", "width": 45, "height": 40, "quantity": 2}],
        "locked_placements": [
            {"item": "A", "x": 0, "y": 0},
            {"item": "A", "x": 50, "y": 0},
        ],
        "max_cuts_per_sheet": 2,
    }
    response = client.post("/plans", json=payload)
    assert response.status_code == 422
    assert "max_cuts_per_sheet" in response.json()["detail"]


def test_max_cuts_per_sheet_locked_within_limit():
    payload = {
        "stock": {"width": 100, "height": 100},
        "kerf": 5,
        "items": [{"id": "A", "width": 45, "height": 40, "quantity": 2}],
        "locked_placements": [
            {"item": "A", "x": 0, "y": 0},
            {"item": "A", "x": 50, "y": 0},
        ],
        "max_cuts_per_sheet": 3,
    }
    response = client.post("/plans", json=payload)
    assert response.status_code == 200
    body = response.json()
    check_plan(body)
    assert body["sheets"][0]["cut_count"] == 3


def test_max_cuts_per_sheet_validation():
    base = {
        "stock": {"width": 100, "height": 100},
        "items": [{"width": 10, "height": 10, "quantity": 1}],
    }
    for bad in (0, -1, 1.5, "two"):
        payload = {**base, "max_cuts_per_sheet": bad}
        assert client.post("/plans", json=payload).status_code == 422


def test_cut_count_returned_per_sheet():
    response = client.post("/plans", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    check_plan(body)  # check_plan asserts cut_count matches the cut tree
    assert all("cut_count" in s for s in body["sheets"])


# ---------------------------------------------------------------------------
# gripper_zone
# ---------------------------------------------------------------------------

GRIPPER_PAYLOAD = {
    "stock": {"width": 100, "height": 100},
    "kerf": 3,
    "items": [{"id": "A", "width": 40, "height": 30, "quantity": 2}],
    "gripper_zone": {"edge": "top", "size": 10},
}


def test_gripper_zone_reserved_as_waste():
    response = client.post("/plans", json=GRIPPER_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    check_plan(body)
    for sheet in body["sheets"]:
        root = sheet["cut_tree"]
        assert root["type"] == "cut" and root["direction"] == "horizontal"
        zone, usable = root["children"]
        assert zone["type"] == "waste"
        assert (zone["x"], zone["y"], zone["width"], zone["height"]) == \
            (0.0, 0.0, 100.0, 10.0)
        assert usable["y"] == 13.0  # size 10 + kerf 3
        for p in sheet["placements"]:
            assert p["y"] >= 13.0


def test_gripper_zone_each_edge():
    expected = {
        "top": ("horizontal", (0.0, 0.0, 100.0, 10.0),
                lambda p: p["y"] >= 13.0),
        "bottom": ("horizontal", (0.0, 90.0, 100.0, 10.0),
                   lambda p: p["y"] + p["height"] <= 87.0),
        "left": ("vertical", (0.0, 0.0, 10.0, 100.0),
                 lambda p: p["x"] >= 13.0),
        "right": ("vertical", (90.0, 0.0, 10.0, 100.0),
                  lambda p: p["x"] + p["width"] <= 87.0),
    }
    for edge, (direction, zone_rect, clear) in expected.items():
        payload = {
            "stock": {"width": 100, "height": 100},
            "kerf": 3,
            "items": [{"id": "A", "width": 40, "height": 30, "quantity": 3}],
            "gripper_zone": {"edge": edge, "size": 10},
        }
        response = client.post("/plans", json=payload)
        assert response.status_code == 200
        body = response.json()
        check_plan(body)
        for sheet in body["sheets"]:
            root = sheet["cut_tree"]
            assert root["type"] == "cut"
            assert root["direction"] == direction
            leaves = list(iter_leaves(root))
            assert any(
                leaf["type"] == "waste"
                and (leaf["x"], leaf["y"], leaf["width"], leaf["height"])
                == zone_rect
                for leaf in leaves
            )
            for p in sheet["placements"]:
                assert clear(p)


def test_gripper_zone_separating_cut_counted():
    # One piece exactly filling the 100x87 usable region: only the
    # gripper cut remains.
    payload = {
        "stock": {"width": 100, "height": 100},
        "kerf": 3,
        "items": [{"id": "A", "width": 100, "height": 87, "quantity": 1}],
        "gripper_zone": {"edge": "top", "size": 10},
    }
    response = client.post("/plans", json=payload)
    assert response.status_code == 200
    body = response.json()
    check_plan(body)
    sheet = body["sheets"][0]
    assert sheet["cut_count"] == 1
    assert sheet["placements"][0]["y"] == 13.0


def test_gripper_zone_counts_towards_max_cuts():
    # A 40x40 piece needs two cuts in the usable region; the gripper cut
    # makes three, exceeding the limit of two.
    payload = {
        "stock": {"width": 100, "height": 100},
        "kerf": 3,
        "items": [{"id": "A", "width": 40, "height": 40, "quantity": 1}],
        "max_cuts_per_sheet": 2,
        "gripper_zone": {"edge": "top", "size": 10},
    }
    response = client.post("/plans", json=payload)
    assert response.status_code == 422
    assert "max_cuts_per_sheet" in response.json()["detail"]

    payload["max_cuts_per_sheet"] = 3
    response = client.post("/plans", json=payload)
    assert response.status_code == 200
    body = response.json()
    check_plan(body)
    assert body["sheets"][0]["cut_count"] == 3


def test_gripper_zone_leaving_no_usable_area_rejected():
    payload = {
        "stock": {"width": 100, "height": 100},
        "kerf": 3,
        "items": [{"id": "A", "width": 10, "height": 10, "quantity": 1}],
        "gripper_zone": {"edge": "top", "size": 98},  # 98 + kerf 3 > 100
    }
    response = client.post("/plans", json=payload)
    assert response.status_code == 422
    assert "gripper zone" in response.json()["detail"]


def test_gripper_zone_item_no_longer_fits_rejected():
    payload = {
        "stock": {"width": 100, "height": 100},
        "kerf": 3,
        "items": [{"id": "poster", "width": 100, "height": 95, "quantity": 1}],
        "gripper_zone": {"edge": "top", "size": 10},
    }
    response = client.post("/plans", json=payload)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "poster" in detail and "usable area" in detail


def test_gripper_zone_validation_rejected():
    base = {
        "stock": {"width": 100, "height": 100},
        "items": [{"width": 10, "height": 10, "quantity": 1}],
    }
    for bad in ({"edge": "top", "size": 0},
                {"edge": "top", "size": -5},
                {"edge": "middle", "size": 10},
                {"edge": "top"},
                {"size": 10},
                "top"):
        payload = {**base, "gripper_zone": bad}
        assert client.post("/plans", json=payload).status_code == 422


def test_gripper_zone_locked_inside_zone_rejected():
    payload = {
        "stock": {"width": 100, "height": 100},
        "kerf": 3,
        "items": [{"id": "A", "width": 40, "height": 40, "quantity": 1}],
        "gripper_zone": {"edge": "top", "size": 10},
        "locked_placements": [{"item": "A", "x": 20, "y": 0}],
    }
    response = client.post("/plans", json=payload)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "#1" in detail and "'A'" in detail
    assert "gripper zone" in detail


def test_gripper_zone_locked_crossing_cut_rejected():
    # y=11 sits in the kerf strip [10, 13) consumed by the separating cut.
    payload = {
        "stock": {"width": 100, "height": 100},
        "kerf": 3,
        "items": [{"id": "A", "width": 40, "height": 40, "quantity": 1}],
        "gripper_zone": {"edge": "top", "size": 10},
        "locked_placements": [{"item": "A", "x": 20, "y": 11}],
    }
    response = client.post("/plans", json=payload)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "#1" in detail and "'A'" in detail
    assert "crosses" in detail


def test_gripper_zone_locked_happy_path():
    payload = {
        "stock": {"width": 100, "height": 100},
        "kerf": 3,
        "items": [{"id": "A", "width": 40, "height": 40, "quantity": 2}],
        "gripper_zone": {"edge": "left", "size": 10},
        "locked_placements": [{"item": "A", "x": 13, "y": 0}],
    }
    response = client.post("/plans", json=payload)
    assert response.status_code == 200
    body = response.json()
    check_plan(body)
    locked = [p for p in body["sheets"][0]["placements"] if p["locked"]]
    assert len(locked) == 1
    assert locked[0]["x"] == 13.0


def test_plan_without_gripper_zone_unchanged():
    # Omitting the field keeps the legacy layout: pieces may be placed
    # anywhere on the sheet, including the strip a gripper zone would
    # reserve.
    payload = {
        "stock": {"width": 100, "height": 100},
        "kerf": 3,
        "items": [{"id": "A", "width": 100, "height": 95, "quantity": 1}],
    }
    response = client.post("/plans", json=payload)
    assert response.status_code == 200
    body = response.json()
    check_plan(body)
    assert body["sheets"][0]["placements"][0]["y"] == 0.0
    # The same request with a top gripper zone of 10 must fail: the
    # piece no longer fits the remaining usable region.
    payload["gripper_zone"] = {"edge": "top", "size": 10}
    assert client.post("/plans", json=payload).status_code == 422
