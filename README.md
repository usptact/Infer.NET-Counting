# Infer.NET Counting (net8.0)

This sample demonstrates Bayesian inference with [Microsoft Infer.NET](https://github.com/dotnet/infer): given repeated draws from an urn with two colours, infer a posterior distribution over the number of balls in the urn.

The model assumes the proportion of colours is known a priori (here 50/50). Each draw selects a ball uniformly at random with replacement. Optionally, observations can be noisy via a colour flip variable.

## Problem

Given a sequence of observed colours from repeated draws, estimate the discrete distribution over the total number of balls. For example, if 10 draws are all blue, how likely is each possible count from 0 to a maximum?

## Solution (model)

- The prior over `numBalls` is uniform over \[0, maxBalls\].
- For each hypothetical ball, its colour is Bernoulli(0.5).
- Each draw indexes a ball uniformly from `0..numBalls-1` and observes its (possibly flipped) colour.
- Inference is performed with `InferenceEngine` to obtain a `Discrete` posterior for `numBalls`.

See `Counting.cs` for the full model.

## Requirements

- .NET 8 SDK

## Build and run

```bash
dotnet build
dotnet run --project Counting.csproj
```

You can tweak the observations in `Counting.cs` to explore different scenarios (e.g., add noise, change number of draws, or increase `maxBalls`).

## Scenarios and interpretation

The program runs several predefined scenarios:

- 10 blue (noiseless): strong evidence for larger feasible counts up to the prior support.
- 5 blue / 5 green (noiseless): evidence spreads more evenly since both colours appear.
- 10 blue (20% noise): the posterior broadens; some unlikely outcomes can be explained by noise.
- Mixed sequence (20% noise): reflects ambiguity due to both colour variety and noise.

The printed posterior is a `Discrete` distribution over counts `0..maxBalls`. Read it as the probability mass for each possible number of balls.

### Notes

- This project targets `net8.0` and uses SDK-style `csproj` with `PackageReference`.
- Packages: `Microsoft.ML.Probabilistic` and `Microsoft.ML.Probabilistic.Compiler`.

## License

Licensed under the Apache License, Version 2.0. See `LICENSE` for details.
