# Minithesis

This is is an incredibly minimal implementation of the core idea of Hypothesis,
mostly intended for demonstration purposes to show someone how they might
get the basics of it up and running.

Minithesis supports:

* Generating arbitrary values inline in the test, including based on previous test results
* Fully generic shrinking
* Preconditions
* A test database
* Targeted Property-Based Testing

And it achieves this all in a relatively small amount of code.

## Notes

* The algorithms for both shrinking and targeting are a bit naive but they're not *terrible* - they should be good enough that using them is better than not having them.
* The database uses Python's DBM module for simplicity. 
* This is probably best read after or in tandem with [our paper about test-case reduction in Hypothesis](https://drmaciver.github.io/papers/reduction-via-generation-preview.pdf)
* This does not necessarily track the core Hypothesis implementation that closely and is more an "in spirit" implementation.
* This probably doesn't work at all well - it has literally one test and I wrote it in about an hour.
* Pull requests to improve clarity *extremely* welcome. It probably won't ever grow many features (I might implement the database at some point) though because you're not supposed to use it in anger.
* I've used the (sadly defunct) Hypothesis-for-Ruby naming conventions because those are better than the Python ones because I actually put some thought into them.
* Currently there is no generator library. I may add a few for illustrative purposes but I have no intention of adding anything like the full range of Hypothesis generators.
