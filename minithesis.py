# This file is part of Minithesis, which may be found at
# https://github.com/DRMacIver/minithesis
#
# This work is copyright (C) 2020 David R. MacIver.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""
This file implements a simple property-based testing library called
minithesis. It's not really intended to be used as is, but is instead
a proof of concept that implements as much of the core ideas of
Hypothesis in a simple way that is designed for people who want to
implement a property-based testing library for non-Python languages.

minithesis is always going to be self-contained in a single file
and consist of < 1000 sloc (as measured by cloc). This doesn't
count comments and I intend to comment on it extensively.


=============
PORTING NOTES
=============

minithesis supports roughly the following features, more or less
in order of most to least important:

1. Test case generation.
2. Test case reduction ("shrinking")
3. A small library of primitive possibilities (generators) and combinators.
4. A Test case database for replay between runs.
5. Targeted property-based testing
6. A caching layer for mapping choice sequences to outcomes


Anything that supports 1 and 2 is a reasonable good first porting
goal. You'll probably want to port most of the possibilities library
because it's easy and it helps you write tests, but don't worry
too much about the specifics.

The test case database is *very* useful and I strongly encourage
you to support it, but if it's fiddly feel free to leave it out
of a first pass.

Targeted property-based testing is very much a nice to have. You
probably don't need it, but it's a rare enough feature that supporting
it gives you bragging rights and who doesn't love bragging rights?

The caching layer you can skip. It's used more heavily in Hypothesis
proper, but in minithesis you only really need it for shrinking
performance, so it's mostly a nice to have.
"""

from __future__ import annotations


import hashlib
import os
from array import array
from enum import IntEnum
from random import Random
from typing import (
    cast,
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Mapping,
    NoReturn,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)


T = TypeVar("T", covariant=True)
S = TypeVar("S", covariant=True)
U = TypeVar("U")  # Invariant

InterestTest = Callable[[array], bool]  # Really array[int] -> bool


class Database(Protocol):
    def __setitem__(self, key: str, value: bytes) -> None:
        ...

    def get(self, key: str) -> Optional[bytes]:
        ...

    def __delitem__(self, key: str) -> None:
        ...


def run_test(
    max_examples: int = 100,
    random: Optional[Random] = None,
    database: Optional[Database] = None,
    quiet: bool = False,
) -> Callable[[Callable[[TestCase], None]], None]:
    """Decorator to run a test. Usage is:

    .. code-block: python

        @run_test()
        def _(test_case):
            n = test_case.choice(1000)
            ...

    The decorated function takes a ``TestCase`` argument,
    and should raise an exception to indicate a test failure.
    It will either run silently or print drawn values and then
    fail with an exception if minithesis finds some test case
    that fails.

    The test will be run immediately, unlike in Hypothesis where
    @given wraps a function to expose it to the the test runner.
    If you don't want it to be run immediately wrap it inside a
    test function yourself.

    Arguments:

    * max_examples: the maximum number of valid test cases to run for.
      Note that under some circumstances the test may run fewer test
      cases than this.
    * random: An instance of random.Random that will be used for all
      nondeterministic choices.
    * database: A dict-like object in which results will be cached and resumed
      from, ensuring that if a test is run twice it fails in the same way.
    * quiet: Will not print anything on failure if True.
    """

    def accept(test: Callable[[TestCase], None]) -> None:
        def mark_failures_interesting(test_case: TestCase) -> None:
            try:
                test(test_case)
            except Exception:
                if test_case.status is not None:
                    raise
                test_case.mark_status(Status.INTERESTING)

        state = TestingState(
            random or Random(), mark_failures_interesting, max_examples
        )

        if database is None:
            # If the database is not set, use a standard cache directory
            # location to persist examples.
            db: Database = DirectoryDB(".minithesis-cache")
        else:
            db = database

        previous_failure = db.get(test.__name__)

        if previous_failure is not None:
            choices = [
                int.from_bytes(previous_failure[i : i + 8], "big")
                for i in range(0, len(previous_failure), 8)
            ]
            state.test_function(TestCase.for_choices(choices))

        if state.result is None:
            state.run()

        if state.valid_test_cases == 0:
            raise Unsatisfiable()

        if state.result is None:
            try:
                del db[test.__name__]
            except KeyError:
                pass
        else:
            db[test.__name__] = b"".join(i.to_bytes(8, "big") for i in state.result)

        if state.result is not None:
            test(TestCase.for_choices(state.result, print_results=not quiet))

    return accept


class TestCase(object):
    """Represents a single generated test case, which consists
    of an underlying set of choices that produce possibilities."""

    @classmethod
    def for_choices(
        cls,
        choices: Sequence[int],
        print_results: bool = False,
    ) -> TestCase:
        """Returns a test case that makes this series of choices."""
        return TestCase(
            prefix=choices,
            random=None,
            max_size=len(choices),
            print_results=print_results,
        )

    def __init__(
        self,
        prefix: Sequence[int],
        random: Optional[Random],
        max_size: float = float("inf"),
        print_results: bool = False,
    ):
        self.prefix = prefix
        # XXX Need a cast because below we assume self.random is not None;
        # it can only be None if max_size == len(prefix)
        self.random: Random = cast(Random, random)
        self.max_size = max_size
        self.choices: array[int] = array("Q")
        self.status: Optional[Status] = None
        self.print_results = print_results
        self.depth = 0
        self.targeting_score: Optional[int] = None

    def choice(self, n: int) -> int:
        """Returns a number in the range [0, n]"""
        result = self.__make_choice(n, lambda: self.random.randint(0, n))
        if self.__should_print():
            print(f"choice({n}): {result}")
        return result

    def weighted(self, p: float) -> int:
        """Return True with probability ``p``."""
        if p <= 0:
            result = self.forced_choice(0)
        elif p >= 1:
            result = self.forced_choice(1)
        else:
            result = bool(self.__make_choice(1, lambda: int(self.random.random() <= p)))
        if self.__should_print():
            print(f"weighted({p}): {result}")
        return result

    def forced_choice(self, n: int) -> int:
        """Inserts a fake choice into the choice sequence, as if
        some call to choice() had returned ``n``. You almost never
        need this, but sometimes it can be a useful hint to the
        shrinker."""
        if n.bit_length() > 64 or n < 0:
            raise ValueError(f"Invalid choice {n}")
        if self.status is not None:
            raise Frozen()
        if len(self.choices) >= self.max_size:
            self.mark_status(Status.OVERRUN)
        self.choices.append(n)
        return n

    def reject(self) -> NoReturn:
        """Mark this test case as invalid."""
        self.mark_status(Status.INVALID)

    def assume(self, precondition: bool) -> None:
        """If this precondition is not met, abort the test and
        mark this test case as invalid."""
        if not precondition:
            self.reject()

    def target(self, score: int) -> None:
        """Set a score to maximize. Multiple calls to this function
        will override previous ones.

        The name and idea come from Löscher, Andreas, and Konstantinos
        Sagonas. "Targeted property-based testing." ISSTA. 2017, but
        the implementation is based on that found in Hypothesis,
        which is not that similar to anything described in the paper.
        """
        self.targeting_score = score

    def any(self, possibility: Possibility[U]) -> U:
        """Return a possible value from ``possibility``."""
        try:
            self.depth += 1
            result = possibility.produce(self)
        finally:
            self.depth -= 1

        if self.__should_print():
            print(f"any({possibility}): {result}")
        return result

    def mark_status(self, status: Status) -> NoReturn:
        """Set the status and raise StopTest."""
        if self.status is not None:
            raise Frozen()
        self.status = status
        raise StopTest()

    def __should_print(self) -> bool:
        return self.print_results and self.depth == 0

    def __make_choice(self, n: int, rnd_method: Callable[[], int]) -> int:
        """Make a choice in [0, n], by calling rnd_method if
        randomness is needed."""
        if n.bit_length() > 64 or n < 0:
            raise ValueError(f"Invalid choice {n}")
        if self.status is not None:
            raise Frozen()
        if len(self.choices) >= self.max_size:
            self.mark_status(Status.OVERRUN)
        if len(self.choices) < len(self.prefix):
            result = self.prefix[len(self.choices)]
        else:
            result = rnd_method()
        self.choices.append(result)
        if result > n:
            self.mark_status(Status.INVALID)
        return result


class Possibility(Generic[T]):
    """Represents some range of values that might be used in
    a test, that can be requested from a ``TestCase``.

    Pass one of these to TestCase.any to get a concrete value.
    """

    def __init__(self, produce: Callable[[TestCase], T], name: Optional[str] = None):
        self.produce = produce
        self.name = produce.__name__ if name is None else name

    def __repr__(self) -> str:
        return self.name

    def map(self, f: Callable[[T], S]) -> Possibility[S]:
        """Returns a ``Possibility`` where values come from
        applying ``f`` to some possible value for ``self``."""
        return Possibility(
            lambda test_case: f(test_case.any(self)),
            name=f"{self.name}.map({f.__name__})",
        )

    def bind(self, f: Callable[[T], Possibility[S]]) -> Possibility[S]:
        """Returns a ``Possibility`` where values come from
        applying ``f`` (which should return a new ``Possibility``
        to some possible value for ``self`` then returning a possible
        value from that."""

        def produce(test_case: TestCase) -> S:
            return test_case.any(f(test_case.any(self)))

        return Possibility[S](
            produce,
            name=f"{self.name}.bind({f.__name__})",
        )

    def satisfying(self, f: Callable[[T], bool]) -> Possibility[T]:
        """Returns a ``Possibility`` whose values are any possible
        value of ``self`` for which ``f`` returns True."""

        def produce(test_case: TestCase) -> T:
            for _ in range(3):
                candidate = test_case.any(self)
                if f(candidate):
                    return candidate
            test_case.reject()

        return Possibility[T](produce, name=f"{self.name}.select({f.__name__})")


def integers(m: int, n: int) -> Possibility[int]:
    """Any integer in the range [m, n] is possible"""
    return Possibility(lambda tc: m + tc.choice(n - m), name=f"integers({m}, {n})")


def lists(
    elements: Possibility[U],
    min_size: int = 0,
    max_size: float = float("inf"),
) -> Possibility[List[U]]:
    """Any lists whose elements are possible values from ``elements`` are possible."""

    def produce(test_case: TestCase) -> List[U]:
        result: List[U] = []
        while True:
            if len(result) < min_size:
                test_case.forced_choice(1)
            elif len(result) + 1 >= max_size:
                test_case.forced_choice(0)
                break
            elif not test_case.weighted(0.9):
                break
            result.append(test_case.any(elements))
        return result

    return Possibility[List[U]](produce, name=f"lists({elements.name})")


def just(value: U) -> Possibility[U]:
    """Only ``value`` is possible."""
    return Possibility[U](lambda tc: value, name=f"just({value})")


def nothing() -> Possibility[NoReturn]:
    """No possible values. i.e. Any call to ``any`` will reject
    the test case."""

    def produce(tc: TestCase) -> NoReturn:
        tc.reject()

    return Possibility(produce)


def mix_of(*possibilities: Possibility[T]) -> Possibility[T]:
    """Possible values can be any value possible for one of ``possibilities``."""
    if not possibilities:
        # XXX Need a cast since NoReturn isn't a T (though perhaps it should be)
        return cast(Possibility[T], nothing())
    return Possibility(
        lambda tc: tc.any(possibilities[tc.choice(len(possibilities) - 1)]),
        name="mix_of({', '.join(p.name for p in possibilities)})",
    )


# XXX This signature requires PEP 646
def tuples(*possibilities: Possibility[Any]) -> Possibility[Any]:
    """Any tuple t of of length len(possibilities) such that t[i] is possible
    for possibilities[i] is possible."""
    return Possibility(
        lambda tc: tuple(tc.any(p) for p in possibilities),
        name="tuples({', '.join(p.name for p in possibilities)})",
    )


# We cap the maximum amount of entropy a test case can use.
# This prevents cases where the generated test case size explodes
# by effectively rejection
BUFFER_SIZE = 8 * 1024


def sort_key(choices: Sequence[int]) -> Tuple[int, Sequence[int]]:
    """Returns a key that can be used for the shrinking order
    of test cases."""
    return (len(choices), choices)


class CachedTestFunction(object):
    """Returns a cached version of a function that maps
    a choice sequence to the status of calling a test function
    on a test case populated with it. Is able to take advantage
    of the structure of the test function to predict the result
    even if exact sequence of choices has not been seen
    previously.

    You can safely omit implementing this at the cost of
    somewhat increased shrinking time.
    """

    def __init__(self, test_function: Callable[[TestCase], None]):
        self.test_function = test_function

        # Tree nodes are either a point at which a choice occurs
        # in which case they map the result of the choice to the
        # tree node we are in after, or a Status object indicating
        # mark_status was called at this point and all future
        # choices are irrelevant.
        #
        # Note that a better implementation of this would use
        # a Patricia trie, which implements long non-branching
        # paths as an array inline. For simplicity we don't
        # do that here.
        # XXX The type of self.tree is recursive
        self.tree: Dict[int, Union[Status, Dict[int, Any]]] = {}

    def __call__(self, choices: Sequence[int]) -> Status:
        # XXX The type of node is problematic
        node: Any = self.tree
        try:
            for c in choices:
                node = node[c]
                # mark_status was called, thus future choices
                # will be ignored.
                if isinstance(node, Status):
                    assert node != Status.OVERRUN
                    return node
            # If we never entered an unknown region of the tree
            # or hit a Status value, then we know that another
            # choice will be made next and the result will overrun.
            return Status.OVERRUN
        except KeyError:
            pass

        # We now have to actually call the test function to find out
        # what happens.
        test_case = TestCase.for_choices(choices)
        self.test_function(test_case)
        assert test_case.status is not None

        # We enter the choices made in a tree.
        node = self.tree
        for i, c in enumerate(test_case.choices):
            if i + 1 < len(test_case.choices) or test_case.status == Status.OVERRUN:
                try:
                    node = node[c]
                except KeyError:
                    node = node.setdefault(c, {})
            else:
                node[c] = test_case.status
        return test_case.status


class TestingState(object):
    def __init__(
        self,
        random: Random,
        test_function: Callable[[TestCase], None],
        max_examples: int,
    ):
        self.random = random
        self.max_examples = max_examples
        self.__test_function = test_function
        self.valid_test_cases = 0
        self.calls = 0
        self.result: Optional[array[int]] = None
        self.best_scoring: Optional[Tuple[int, Sequence[int]]] = None
        self.test_is_trivial = False

    def test_function(self, test_case: TestCase) -> None:
        try:
            self.__test_function(test_case)
        except StopTest:
            pass
        if test_case.status is None:
            test_case.status = Status.VALID
        self.calls += 1
        if test_case.status >= Status.INVALID and len(test_case.choices) == 0:
            self.test_is_trivial = True
        if test_case.status >= Status.VALID:
            self.valid_test_cases += 1

            if test_case.targeting_score is not None:
                relevant_info = (test_case.targeting_score, test_case.choices)
                if self.best_scoring is None:
                    self.best_scoring = relevant_info
                else:
                    best, _ = self.best_scoring
                    if test_case.targeting_score > best:
                        self.best_scoring = relevant_info

        if test_case.status == Status.INTERESTING and (
            self.result is None or sort_key(test_case.choices) < sort_key(self.result)
        ):
            self.result = test_case.choices

    def target(self) -> None:
        """If any test cases have had ``target()`` called on them, do a simple
        hill climbing algorithm to attempt to optimise that target score."""
        if self.result is not None or self.best_scoring is None:
            return

        def adjust(i: int, step: int) -> bool:
            """Can we improve the score by changing choices[i] by ``step``?"""
            assert self.best_scoring is not None
            score, choices = self.best_scoring
            if choices[i] + step < 0 or choices[i].bit_length() >= 64:
                return False
            attempt = array("Q", choices)
            attempt[i] += step
            test_case = TestCase(
                prefix=attempt, random=self.random, max_size=BUFFER_SIZE
            )
            self.test_function(test_case)
            assert test_case.status is not None
            return (
                test_case.status >= Status.VALID
                and test_case.targeting_score is not None
                and test_case.targeting_score > score
            )

        while self.should_keep_generating():
            i = self.random.randrange(0, len(self.best_scoring[1]))
            sign = 0
            for k in [1, -1]:
                if not self.should_keep_generating():
                    return
                if adjust(i, k):
                    sign = k
                    break
            if sign == 0:
                continue

            k = 1
            while self.should_keep_generating() and adjust(i, sign * k):
                k *= 2

            while k > 0:
                while self.should_keep_generating() and adjust(i, sign * k):
                    pass
                k //= 2

    def run(self) -> None:
        self.generate()
        self.target()
        self.shrink()

    def should_keep_generating(self) -> bool:
        return (
            not self.test_is_trivial
            and self.result is None
            and self.valid_test_cases < self.max_examples
            and
            # We impose a limit on the maximum number of calls as
            # well as the maximum number of valid examples. This is
            # to avoid taking a prohibitively long time on tests which
            # have hard or impossible to satisfy preconditions.
            self.calls < self.max_examples * 10
        )

    def generate(self) -> None:
        """Run random generation until either we have found an interesting
        test case or hit the limit of how many test cases we should
        evaluate."""
        while self.should_keep_generating() and (
            self.best_scoring is None or self.valid_test_cases <= self.max_examples // 2
        ):
            self.test_function(
                TestCase(prefix=(), random=self.random, max_size=BUFFER_SIZE)
            )

    def shrink(self) -> None:
        """If we have found an interesting example, try shrinking it
        so that the choice sequence leading to our best example is
        shortlex smaller than the one we originally found. This improves
        the quality of the generated test case, as per our paper.

        https://drmaciver.github.io/papers/reduction-via-generation-preview.pdf
        """
        if not self.result:
            return

        # Shrinking will typically try the same choice sequences over
        # and over again, so we cache the test function in order to
        # not end up reevaluating it in those cases. This also allows
        # us to catch cases where we try something that is e.g. a prefix
        # of something we've previously tried, which is guaranteed
        # not to work.
        cached = CachedTestFunction(self.test_function)

        def consider(choices: array[int]) -> bool:
            if choices == self.result:
                return True
            return cached(choices) == Status.INTERESTING

        assert consider(self.result)

        # We are going to perform a number of transformations to
        # the current result, iterating until none of them make any
        # progress - i.e. until we make it through an entire iteration
        # of the loop without changing the result.
        prev = None
        while prev != self.result:
            prev = self.result

            # A note on weird loop order: We iterate backwards
            # through the choice sequence rather than forwards,
            # because later bits tend to depend on earlier bits
            # so it's easier to make changes near the end and
            # deleting bits at the end may allow us to make
            # changes earlier on that we we'd have missed.
            #
            # Note that we do not restart the loop at the end
            # when we find a successful shrink. This is because
            # things we've already tried are less likely to work.
            #
            # If this guess is wrong, that's OK, this isn't a
            # correctness problem, because if we made a successful
            # reduction then we are not at a fixed point and
            # will restart the loop at the end the next time
            # round. In some cases this can result in performance
            # issues, but the end result should still be fine.

            # First try deleting each choice we made in chunks.
            self.shrink_remove(consider)

            # Now we try replacing blocks of choices with zeroes.
            self.result = shrink_zeroes(self.result, consider)

            # Now try replacing each choice with a smaller value
            self.result = shrink_lower(self.result, consider)


            # NB from here on this is just showing off cool shrinker tricks and
            # you probably don't need to worry about it and can skip these bits
            # unless they're easy and you want bragging rights for how much
            # better you are at shrinking than the local QuickCheck equivalent.

            # Try sorting out of order ranges of choices, as ``sort(x) <= x``,
            # so this is always a lexicographic reduction.
            k = 8
            while k > 1:
                for i in range(len(self.result) - k - 1, -1, -1):
                    consider(
                        self.result[:i]
                        + array("Q", sorted(self.result[i : i + k]))
                        + self.result[i + k :]
                    )
                k //= 2

            # Try adjusting nearby pairs of integers by redistributing value
            # between them. This is useful for tests that depend on the
            # sum of some generated values.
            for k in [2, 1]:
                for i in range(len(self.result) - 1 - k, -1, -1):
                    j = i + k
                    # This check is necessary because the previous changes
                    # might have shrunk the size of result, but also it's tedious
                    # to write tests for this so I didn't.
                    if j < len(self.result):  # pragma: no cover
                        # Try swapping out of order pairs
                        if self.result[i] > self.result[j]:
                            is_interesting_with_replacement(
                                self.result,
                                {j: self.result[i], i: self.result[j]},
                                consider,
                            )
                        # j could be out of range if the previous swap succeeded.
                        if j < len(self.result) and self.result[i] > 0:
                            previous_i = self.result[i]
                            previous_j = self.result[j]
                            bin_search_down(
                                0,
                                previous_i,
                                lambda v: is_interesting_with_replacement(
                                    self.result,
                                    {i: v, j: previous_j + (previous_i - v)},
                                    consider,
                                ),
                            )

    def shrink_remove(self, consider):
        # Try removing chunks, starting from the end.
        # We try longer chunks because this allows us to
        # delete whole composite elements: e.g. deleting an
        # element from a generated list requires us to delete
        # both the choice of whether to include it and also
        # the element itself, which may involve more than one
        # choice. Some things will take more than 8 choices
        # in the sequence. That's too bad, we may not be
        # able to delete those. In Hypothesis proper we
        # record the boundaries corresponding to ``any``
        # calls so that we can try deleting those, but
        # that's pretty high overhead and also a bunch of
        # slightly annoying code that it's not worth porting.
        #
        # We could instead do a quadratic amount of work
        # to try all boundaries, but in general we don't
        # want to do that because even a shrunk test case
        # can involve a relatively large number of choices.
        for n_to_remove in range(8, 0, -1):
            removal_index = len(self.result) - n_to_remove - 1
            while removal_index >= 0:
                if removal_index >= len(self.result):
                    # Can happen if we successfully lowered
                    # the value at removal_index - 1
                    removal_index -= 1
                    continue
                attempt = (
                    self.result[:removal_index]
                    + self.result[removal_index + n_to_remove :]
                )
                assert len(attempt) < len(self.result)
                if not consider(attempt):
                    # If we have dependencies on some length
                    # parameter, e.g. draw a number between
                    # 0 and 10 and then draw that many
                    # elements, shrinking often gets stuck
                    # because the decision to add many
                    # elements was made early in the chain.
                    # We check if the element just
                    # prior to our removal could be a length
                    # and try decreasing it.
                    # This can't delete everything that occurs
                    # as described, but it can delete some
                    # things and often will get us unstuck
                    # when nothing else does.
                    if removal_index > 0 and attempt[removal_index - 1] > 0:
                        attempt[removal_index - 1] -= 1
                        # If successful, retry the removal pass
                        if consider(attempt):
                            continue
                    removal_index -= 1


def shrink_zeroes(current: array[int], test: InterestTest) -> array[int]:
    # Try zero-ing out sections.
    # Note that we skip a block of size 1 because that will
    # be taken care of by a pass that tries lower values.
    # Often (but not always), a block of all zeroes is the
    # shortlex smallest value that a region can be.
    for size in [8, 4, 2]:
        i = len(current) - size
        while i >= 0:
            # Zero out section starting at i
            attempt = (
                current[:i] + array("Q", (0 for _ in range(size))) + current[i + size :]
            )

            if test(attempt):
                current = attempt
                # If we've succeeded then all of [i, i + size]
                # is zero so we adjust i so that the next region
                # does not overlap with this at all.
                i -= size
            else:
                # Otherwise we might still be able to zero some
                # of these values but not the last one, so we
                # just go back one.
                i -= 1
    return current


def shrink_lower(current: array[int], is_interesting: InterestTest) -> array[int]:
    # Try replacing each choice with a smaller value
    # by doing a binary search. This will replace n with 0 or n - 1
    # if possible, but will also more efficiently replace it with
    # a smaller number than doing multiple subtractions would.
    for i in reversed(range(len(current))):
        current[i] = bin_search_down(
            0,
            current[i],
            lambda v: is_interesting_with_replacement(current, {i: v}, is_interesting),
        )
    return current


def is_interesting_with_replacement(
    current: array[int], values: Mapping[int, int], test: InterestTest
) -> bool:
    """Attempts to replace some indices in the current
    result with new values. Useful for some purely lexicographic
    reductions that we are about to perform."""
    assert current is not None

    # If replacement map is out-of-range, abort.
    # Some other shrinking pass is probably better.
    if any(i >= len(current) for i in values.keys()):
        return False

    attempt = array("Q", current)
    for i, v in values.items():
        attempt[i] = v
    return test(attempt)


def bin_search_down(lo: int, hi: int, f: Callable[[int], bool]) -> int:
    """Returns n in [lo, hi] such that f(n) is True,
    where it is assumed and will not be checked that
    f(hi) is True.

    Will return ``lo`` if ``f(lo)`` is True, otherwise
    the only guarantee that is made is that ``f(n - 1)``
    is False and ``f(n)`` is True. In particular this
    does *not* guarantee to find the smallest value,
    only a locally minimal one.
    """
    if f(lo):
        return lo
    while lo + 1 < hi:
        mid = lo + (hi - lo) // 2
        if f(mid):
            hi = mid
        else:
            lo = mid
    return hi


class DirectoryDB:
    """A very basic key/value store that just uses a file system
    directory to store values. You absolutely don't have to copy this
    and should feel free to use a more reasonable key/value store
    if you have easy access to one."""

    def __init__(self, directory: str):
        self.directory = directory
        try:
            os.mkdir(directory)
        except FileExistsError:
            pass

    def __to_file(self, key: str) -> str:
        return os.path.join(
            self.directory, hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
        )

    def __setitem__(self, key: str, value: bytes) -> None:
        with open(self.__to_file(key), "wb") as o:
            o.write(value)

    def get(self, key: str) -> Optional[bytes]:
        f = self.__to_file(key)
        if not os.path.exists(f):
            return None
        with open(f, "rb") as i:
            return i.read()

    def __delitem__(self, key: str) -> None:
        try:
            os.unlink(self.__to_file(key))
        except FileNotFoundError:
            raise KeyError()


class Frozen(Exception):
    """Attempted to make choices on a test case that has been
    completed."""


class StopTest(Exception):
    """Raised when a test should stop executing early."""


class Unsatisfiable(Exception):
    """Raised when a test has no valid examples."""


class Status(IntEnum):
    # Test case didn't have enough data to complete
    OVERRUN = 0

    # Test case contained something that prevented completion
    INVALID = 1

    # Test case completed just fine but was boring
    VALID = 2

    # Test case completed and was interesting
    INTERESTING = 3
