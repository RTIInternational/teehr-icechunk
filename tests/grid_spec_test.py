"""window_hours: the hours each lead of a variable covers."""
from teehr_icechunk.grib_index import window
from teehr_icechunk.grid_spec import window_hours

APCP = {"cell_methods": "time: sum", "window_reset_hours": 6}


def test_windowed_alternates_3_and_6_hours_at_0p25():
    hours = window_hours(range(3, 241, 3), APCP)
    assert [hours[lead] for lead in (3, 6, 9, 12, 237, 240)] == [3, 6, 3, 6, 3, 6]
    assert window_hours(range(3, 241, 3), {"cell_methods": "time: maximum", "window_reset_hours": 6}) == hours


def test_windowed_is_6_hours_at_0p5():
    assert set(window_hours(range(246, 841, 6), APCP).values()) == {6}


def test_matches_the_index_labels():
    # The window each lead covers is the one its .idx message is labelled with
    for lead, hours in window_hours(range(3, 241, 3), APCP).items():
        assert window(lead, "acc", 6) == f"{lead - hours}-{lead} hour acc fcst"


def test_other_variables_cover_the_lead_spacing():
    # Instantaneous, or a window method without a reset (e.g. dynamical's averages since the previous step)
    leads = [0, 3, 6, 240, 246]
    expected = {3: 3, 6: 3, 240: 234, 246: 6}
    assert window_hours(leads, {"cell_methods": "time: point"}) == expected
    assert window_hours(leads, {"cell_methods": "area: mean time: mean (interval: 3 hours)"}) == expected
    assert window_hours(leads, {}) == expected


def test_windows_that_never_reset_cover_the_hours_since_the_cycle():
    # e.g. an accumulation from the cycle: 0-1, 0-2, ... hours
    assert window_hours([0, 1, 2, 18], {"cell_methods": "time: sum", "window_reset_hours": 0}) == {1: 1, 2: 2, 18: 18}
    assert window(18, "acc", 0) == "0-18 hour acc fcst"
