import gzip
import json

import pytest

from swe_smith.probes.archive_json import iter_json_archives, read_json_archive


def test_atomic_compression_overlap_and_unlink_race(tmp_path):
    path = tmp_path / 'one.json'
    value = {'text': '中文', 'step': 0}
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
    with gzip.open(path.with_name(path.name + '.gz'), 'wt', encoding='utf-8') as out:
        json.dump(value, out, ensure_ascii=False)
    listed = iter_json_archives(tmp_path)
    assert listed == [path]
    assert read_json_archive(path) == value
    path.unlink()
    assert read_json_archive(listed[0]) == value
    assert read_json_archive(iter_json_archives(tmp_path)[0]) == value


@pytest.mark.parametrize('payload', [b'not gzip', gzip.compress(b'{"step":0}')[:-6]])
def test_corruption_is_not_silently_replaced_by_empty_data(tmp_path, payload):
    path = tmp_path / 'bad.json.gz'
    path.write_bytes(payload)
    with pytest.raises((OSError, EOFError)):
        read_json_archive(path)
