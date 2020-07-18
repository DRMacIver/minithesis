from minithesis import (
    run_test,
    Possibility,
    Unsatisfiable,
    Status,
    TestingState as State,
    CachedTestFunction,
)

import pytest
from random import Random


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


def test_test_cases_satisfy_preconditions():
    @run_test()
    def _(test_case):
        n = test_case.choice(10)
        test_case.assume(n != 0)
        assert n != 0


def test_error_on_too_strict_precondition():
    with pytest.raises(Unsatisfiable):

        @run_test()
        def _(test_case):
            n = test_case.choice(10)
            test_case.reject()


def test_function_cache():
    def tf(tc):
        if tc.choice(1000) >= 200:
            tc.mark_status(Status.INTERESTING)
        if tc.choice(1) == 0:
            tc.reject()

    state = State(Random(0), tf, 100)

    cache = CachedTestFunction(state.test_function)

    assert cache([1, 1]) == Status.VALID
    assert cache([1]) == Status.OVERRUN
    assert cache([1000]) == Status.INTERESTING
    assert cache([1000]) == Status.INTERESTING
    assert cache([1000, 1]) == Status.INTERESTING

    assert state.calls == 2
