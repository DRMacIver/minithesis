# Minithesis

This is is an incredibly minimal implementation of the core idea of Hypothesis,
mostly intended for demonstration purposes to show someone how they might
get the basics of it up and running.

The goal is for this to always be a single file implementation in < 1000 sloc (i.e. 1000 lines of code not counting blanks, comments, or docstrings).
Tests and demo code are not counted towards that budget. Currently it is significantly smaller than that.

## Notes

* This is probably best read after or in tandem with [our paper about test-case reduction in Hypothesis](https://drmaciver.github.io/papers/reduction-via-generation-preview.pdf)
* This does not necessarily track the core Hypothesis implementation that closely and is more an "in spirit" implementation.
* This probably doesn't work at all well - it has literally one test and I wrote it in about an hour.
* Pull requests to improve clarity *extremely* welcome. It probably won't ever grow many features (I might implement the database at some point) though because you're not supposed to use it in anger.
* I've used the (sadly defunct) Hypothesis-for-Ruby naming conventions because those are better than the Python ones because I actually put some thought into them.


## Missing Features

Here are some things that I might implement if I can fit them into the budget but currently haven't:

* Targeted property-based testing
