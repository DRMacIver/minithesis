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
        @run_test()
        def _(test_case):
            ls = test_case.any(list_of_integers)
            assert sum(ls) <= 1000

    captured = capsys.readouterr()

    assert captured.out.strip() == "any(list_of_integers): [1001]"
