from minithesis import run_test, Possibility

import pytest


@Possibility
def list_of_integers(test_case):
    result = []
    while test_case.weighted(0.9):
        result.append(test_case.choice(10000))
    return result


def test_finds_small_list(capsys):

    with pytest.raises(AssertionError):
        @run_test(database={})
        def _(test_case):
            ls = test_case.any(list_of_integers)
            assert sum(ls) <= 1000

    captured = capsys.readouterr()

    assert captured.out.strip() == "any(list_of_integers): [1001]"


def test_reuses_results_from_the_database():
    db = {}
    count = 0

    def run():
        with pytest.raises(AssertionError):
            @run_test(database=db)
            def _(test_case):
                nonlocal count
                count += 1
                assert test_case.choice(10000) < 10

    run()

    assert len(db) == 1
    prev_count = count

    run()

    assert len(db) == 1
    assert count == prev_count + 2

