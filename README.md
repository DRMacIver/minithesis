# Minithesis

This is is an incredibly minimal implementation of the core idea of [Hypothesis](https://github.com/HypothesisWorks/hypothesis),
mostly intended for demonstration purposes to show someone how they might
get the basics of it up and running.

Minithesis supports:

* Generating arbitrary values inline in the test, including based on previous test results
* Fully generic shrinking
* Preconditions
* A test database
* Targeted Property-Based Testing

And it achieves this all in a relatively small amount of code - my original goal was to keep it under 1000 sloc, but then I implemented about twice the feature set I'd intended to and it didn't hit 300 sloc,
so I doubt I'll come close to that (including comments, docstrings, etc. it might eventually hit 1kloc as I try to make it more easily understandable and more thoroughly explained)

## Notes

* The algorithms for both shrinking and targeting are a bit naive but they're not *terrible* - they should be good enough that using them is better than not having them.
* The database uses Python's DBM module for simplicity. 
* This is probably best read after or in tandem with [our paper about test-case reduction in Hypothesis](https://drmaciver.github.io/papers/reduction-via-generation-preview.pdf)
* This does not necessarily track the core Hypothesis implementation that closely and is more an "in spirit" implementation.
* This probably doesn't work all that well - it's tolerably well tested, but nobody has ever used it for real and probably nobody ever will because why would they when Hypothesis exists?
* Pull requests to improve clarity *extremely* welcome. It probably won't ever grow many features (I might implement the database at some point) though because you're not supposed to use it in anger.
* I've used the (sadly defunct) Hypothesis-for-Ruby naming conventions because those are better than the Python ones because I actually put some thought into them.
* There is a fairly minimal generator library just to get you started but it's nothing resembling comprehensive and is unlikely to ever be.


## Minithesis Ports

There are a number of ports of minithesis (:tada:). The following are the ones I'm aware of:

* Martin Janiczek's Elm port, [elm-minithesis](https://github.com/Janiczek/elm-minithesis)
* Jack Firth's racket port, [miniracksis](https://github.com/jackfirth/miniracksis/)
* Amanda Walker's Haskell port, [haskell-minithesis](https://github.com/AnOctopus/haskell-minithesis)
* Dmitry Dygalo and Rik de Kort's Rust port, [minithesis-rust](https://github.com/Rik-de-Kort/minithesis-rust)
* Justin Blank's Java port, [jiminy-thesis](https://github.com/hyperpape/jiminy-thesis)

If you write a port, please submit a pull request to add it to the list!
