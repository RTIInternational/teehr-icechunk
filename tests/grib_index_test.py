"""GRIB2 index parsing, on indexes of the GEFS v12 reforecast (2019-12-25, a weekly 11-member cycle)."""
from pathlib import Path

import pytest

from teehr_icechunk.grib_index import (
    StaleIndexError, check_last_message, message_ranges, parse_index, window,
)

FIXTURES = Path(__file__).parent / "fixtures" / "gefs"
APCP_1_10 = (FIXTURES / "apcp_sfc_2019122500_c00_Days_1-10.grib2.idx").read_text()
APCP_10_35 = (FIXTURES / "apcp_sfc_2019122500_p07_Days_10-35.grib2.idx").read_text()
TMP_1_10 = (FIXTURES / "tmp_2m_2019122500_c00_Days_1-10.grib2.idx").read_text()
SIZES = {"apcp_1_10": 23_892_815, "tmp_1_10": 59_782_340, "apcp_10_35": 9_908_268}


def test_parse_reads_every_message():
    messages = parse_index(APCP_1_10)
    assert len(messages) == 80
    first = messages[0]
    assert (first.start, first.date, first.element, first.level, first.window) == (
        0, "2019122500", "APCP", "surface", "0-3 hour acc fcst")
    assert first.selectors == ("ENS=low-res ctl",)
    assert parse_index(APCP_10_35)[-1].selectors == ("ENS=+7",)


def test_windows_match_the_index_labels():
    """APCP buckets reset every 6 h (0-3, 0-6, 6-9, 6-12, ...); beyond day 10 each is 6 h; TMP is instantaneous."""
    assert [window(lead, "acc", 6) for lead in (3, 6, 9, 12)] == [m.window for m in parse_index(APCP_1_10)[:4]]
    assert [m.window for m in parse_index(APCP_1_10)] == [window(lead, "acc", 6) for lead in range(3, 241, 3)]
    assert [m.window for m in parse_index(APCP_10_35)] == [window(lead, "acc", 6) for lead in range(246, 841, 6)]
    assert [m.window for m in parse_index(TMP_1_10)] == [window(lead, "instant") for lead in range(3, 241, 3)]


def test_ranges_end_at_the_next_message_and_the_last_at_the_file_end():
    ranges = message_ranges(parse_index(APCP_1_10), SIZES["apcp_1_10"])
    assert ranges[0][1:] == (0, 226_323)
    assert sum(length for *_, length in ranges) == SIZES["apcp_1_10"]


def test_ranges_that_do_not_fit_the_file_are_refused():
    with pytest.raises(StaleIndexError):
        message_ranges(parse_index(APCP_1_10), 1_000_000)  # the file is smaller than the index says
    with pytest.raises(StaleIndexError):
        message_ranges([], 10)


def test_last_message_header_must_fit():
    header = b"GRIB" + b"\x00\x00\x00\x02" + (500).to_bytes(8, "big")
    check_last_message(header, offset=1_000, file_size=1_500)
    with pytest.raises(StaleIndexError, match="past"):
        check_last_message(header, offset=1_100, file_size=1_500)
    with pytest.raises(StaleIndexError, match="GRIB2"):
        check_last_message(b"GRIB\x00\x00\x00\x01" + bytes(8), offset=0, file_size=10)
    with pytest.raises(StaleIndexError, match="GRIB2"):
        check_last_message(b"\x00" * 16, offset=0, file_size=10)
