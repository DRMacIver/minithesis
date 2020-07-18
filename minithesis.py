"""
This file implements a simple property-based testing library called
minithesis. It's not really intended to be used as is, but is instead
a proof of concept that implements as much of the core ideas of
Hypothesis in a simple way that is designed for people who want to
implement a property-based testing library for non-Python languages.

minithesis is always going to be self-contained in a single file
and consist of < 1000 sloc (as measured by cloc). This doesn't
count comments and I intend to comment on it extensively.

"""


from array import array
from enum import IntEnum
from random import Random
import dbm


def run_test(max_examples=100, seed=None, database=None):
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
    * seed: A fixed seed to use for randomness.
    * dict: A dict-like object in which results will be cached and resumed
      from, ensuring that if a test is run twice it fails in the same way.
    """

    def accept(test):
        random = Random(seed)

        def mark_failures_interesting(test_case):
            try:
                test(test_case)
            except Exception:
                if test_case.status is not None:
                    raise
                test_case.mark_status(Status.INTERESTING)

        state = TestingState(random, mark_failures_interesting, max_examples)

        if database is None:
            # We're using the DBM module because it's an easy default.
            # We don't use this in real Hypothesis - we've got a weird
            # thing there designed to be checked into git but honestly
            # nobody ever checks it into git - and I would encourage you
            # to use some more sensible key/value store here.
            db = dbm.open(".minithesis-cache", "c")
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

        if state.result is None:
            db.pop(test.__name__, None)
        else:
            db[test.__name__] = b"".join(i.to_bytes(8, "big") for i in state.result)

        if hasattr(db, "close"):
            db.close()

        if state.result is not None:
            test(TestCase.for_choices(state.result, print_results=True))

    return accept


class Possibility(object):
    """Represents some range of values that might be used in
    a test, that can be requested from a ``TestCase``.

    Pass one of these to TestCase.any to get a concrete value.
    """

    def __init__(self, produce=None):
        if produce is not None:
            self.produce = produce
            self.name = produce.__name__
        else:
            self.name = "..."

    def __repr__(self):
        if self.name is None:
            return "Possibility(...)"
        else:
            return self.name

    def produce(self, source):
        raise NotImplementedError()


class TestCase(object):
    """Represents a single generated test case, which consists
    of an underlying set of choices that produce possibilities."""

    @classmethod
    def for_choices(cls, choices, print_results=False):
        """Returns a test case that makes this series of choices."""
        return TestCase(
            prefix=choices,
            random=None,
            max_size=len(choices),
            print_results=print_results,
        )

    def __init__(self, prefix, random, max_size=float("inf"), print_results=False):
        self.prefix = prefix
        self.random = random
        self.max_size = max_size
        self.choices = array("I")
        self.status = None
        self.print_results = print_results
        self.depth = 0

    def choice(self, n):
        """Returns a number in the range [0, n]"""
        result = self.__make_choice(n, lambda: self.random.randint(0, n))
        if self.__should_print():
            print(f"choice({n}): {result}")
        return result

    def weighted(self, p):
        """Return True with probability ``p``."""
        result = bool(self.__make_choice(1, lambda: int(self.random.random() <= p)))
        if self.__should_print():
            print(f"weighted({p}): {result}")
        return result

    def reject(self):
        """Mark this test case as invalid."""
        self.mark_status(Status.INVALID)

    def assume(self, precondition):
        """If this precondition is not met, abort the test and
        mark this test case as invalid."""
        if not precondition:
            self.reject()

    def any(self, possibility):
        """Return a possible value from ``possibility``."""
        try:
            self.depth += 1
            result = possibility.produce(self)
        finally:
            self.depth -= 1

        if self.__should_print():
            print(f"any({possibility}): {result}")
        return result

    def mark_status(self, status):
        """Set the status and raise StopTest."""
        if self.status is not None:
            raise Frozen()
        self.status = status
        raise StopTest()

    def __should_print(self):
        return self.print_results and self.depth == 0

    def __make_choice(self, n, rnd_method):
        """Make a choice in [0, n], by calling rnd_method if
        randomness is needed."""
        if n.bit_length() >= 64 or n < 0:
            raise ValueError(f"Invalid choice {n}")
        if self.status is not None:
            raise Frozen()
        if len(self.choices) >= self.max_size:
            self.mark_status(Status.OVERRUN)
        if len(self.choices) < len(self.prefix):
            result = self.prefix[len(self.choices)]
        else:
            result = rnd_method()
        if result > n:
            self.mark_status(Status.INVALID)
        self.choices.append(result)
        return result


# We cap the maximum amount of entropy a test case can use.
# This prevents cases where the generated test case size explodes
# by effectively rejection
BUFFER_SIZE = 8 * 1024


def sort_key(choices):
    """Returns a key that can be used for the shrinking order
    of test cases."""
    return (len(choices), choices)


class TestingState(object):
    def __init__(self, random, test_function, max_examples):
        self.random = random
        self.max_examples = max_examples
        self.__test_function = test_function
        self.valid_test_cases = 0
        self.calls = 0
        self.result = None

    def test_function(self, test_case):
        try:
            self.__test_function(test_case)
            if test_case.status is None:
                test_case.status = Status.VALID
        except StopTest:
            pass
        self.calls += 1
        if test_case.status >= Status.VALID:
            self.valid_test_cases += 1
        if test_case.status == Status.INTERESTING:
            if self.result is None or sort_key(test_case.choices) < sort_key(
                self.result
            ):
                self.result = test_case.choices

    def run(self):
        self.generate()
        self.shrink()

    def generate(self):
        while (
            self.result is None
            and self.valid_test_cases < self.max_examples
            and self.calls < self.max_examples * 10
        ):
            self.test_function(
                TestCase(prefix=(), random=self.random, max_size=BUFFER_SIZE)
            )

    def shrink(self):
        if self.result is None:
            return

        def consider(choices):
            if choices == self.result:
                return True
            tc = TestCase.for_choices(choices)
            self.test_function(tc)
            return tc.status == Status.INTERESTING

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

            # First try deleting each choice we made in chunks
            k = 8
            while k > 0:
                i = len(self.result) - k - 1
                while i >= 0:
                    attempt = self.result[:i] + self.result[i + k :]
                    assert len(attempt) < len(self.result)
                    if not consider(attempt):
                        i -= 1
                k //= 2

            # Now try replacing blocks of choices with zeroes
            k = 8
            while k > 0:
                i = len(self.result) - k - 1
                while i >= 0:
                    attempt = (
                        self.result[:i] + array("I", [0] * k) + self.result[i + k :]
                    )
                    if consider(attempt):
                        i -= k
                    else:
                        i -= 1
                k //= 2

            # Now try replacing each choice with a smaller value
            # by doing a binary search.
            i = len(self.result) - 1
            while i >= 0:
                # We assume that if we could replace the choice with zero
                # then we would have on the previous step. Strictly
                # this needn't be true.
                lo = 0
                hi = self.result[i]
                while lo + 1 < hi:
                    mid = lo + (hi - lo) // 2
                    attempt = array("I", self.result)
                    attempt[i] = mid
                    if consider(attempt):
                        hi = mid
                    else:
                        lo = mid
                i -= 1


class Frozen(Exception):
    """Attempted to make choices on a test case that has been
    completed."""


class StopTest(Exception):
    """Raised when a test should stop executing early."""


class Status(IntEnum):
    # Test case didn't have enough data to complete
    OVERRUN = 0

    # Test case contained something that prevented completion
    INVALID = 1

    # Test case completed just fine but was boring
    VALID = 2

    # Test case completed and was interesting
    INTERESTING = 3
