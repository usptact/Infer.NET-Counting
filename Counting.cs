using System;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;
using ModelRange = Microsoft.ML.Probabilistic.Models.Range;

//
// Urn Counting with Infer.NET
// ----------------------------
// This model estimates the (discrete) probability distribution over the number of balls
// in an urn given a sequence of observed colours from draws with replacement.
// The observations can be noisy, governed by a "switchedColor" random variable
// that flips the observed colour with some probability (noise rate).
//
// Intuition:
// - If you observe many blue draws with no noise, higher counts are typically more plausible
//   up to a point, with the posterior shaped by the combinatorics of uniform indexing.
// - Introducing noise broadens the posterior since mismatches can be explained by noise.
//

namespace Counting
{
	class Program
	{
		static void Main(string[] args)
		{
			// Each scenario defines: a name, the observed sequence, a maximum number of balls to consider,
			// and a noise rate (probability of flipping the colour during observation).
			var scenarios = new (string name, bool[] data, int maxBalls, double noise)[]
			{
				("10 blue, noiseless", new bool[]{ true, true, true, true, true, true, true, true, true, true }, 8, 0.0),
				("5 blue / 5 green, noiseless", new bool[]{ true, true, true, true, true, false, false, false, false, false }, 8, 0.0),
				("10 blue, 20% noise", new bool[]{ true, true, true, true, true, true, true, true, true, true }, 8, 0.2),
				("Mixed, 20% noise", new bool[]{ true, false, true, false, true, true, false, true, false, true }, 8, 0.2)
			};

			foreach (var scenario in scenarios)
			{
				Console.WriteLine("=== Scenario: " + scenario.name + " (maxBalls=" + scenario.maxBalls + ", noise=" + scenario.noise + ") ===");
				Console.WriteLine("Observed sequence: [" + string.Join(", ", Array.ConvertAll(scenario.data, b => b ? "blue" : "green")) + "]");

				// Variables describing the population
				int maxBalls = scenario.maxBalls;
				ModelRange ball = new ModelRange(maxBalls+1); // so that numBalls = (0,...,maxBalls)
				Variable<int> numBalls = Variable.DiscreteUniform(ball);
				VariableArray<bool> isBlue = Variable.Array<bool>(ball);
				isBlue[ball] = Variable.Bernoulli(0.5).ForEach(ball);

				// Variables describing the observations
				ModelRange draw = new ModelRange(scenario.data.Length);
				VariableArray<bool> observedBlue = Variable.Array<bool>(draw);
				VariableArray<int> ballIndex = Variable.Array<int>(draw);
				using (Variable.ForEach(draw))
				{
					ballIndex[draw] = Variable.DiscreteUniform(numBalls);
					Variable<bool> switchedColor = Variable.Bernoulli(scenario.noise);
					using (Variable.Switch(ballIndex[draw])) {
						using (Variable.If(switchedColor))
						{
							observedBlue[draw] = !isBlue[ballIndex[draw]];
						}
						using (Variable.IfNot(switchedColor))
						{
							observedBlue[draw] = isBlue[ballIndex[draw]];
						}
					}
				}

				observedBlue.ObservedValue = scenario.data;

				// Inference
				InferenceEngine engine = new InferenceEngine();
				// Posterior over numBalls: Discrete distribution over 0..maxBalls
				Discrete numberOfBalls = engine.Infer<Discrete>( numBalls );
				Console.WriteLine("Posterior over number of balls: " + numberOfBalls);
				Console.WriteLine("(probabilities correspond to counts 0.." + scenario.maxBalls + ")");
				Console.WriteLine();
			}

			// Answer key
			// ----------
			// 10 blue, without noise:
			// numBalls = Discrete(0 0.5079 0.3097 0.09646 0.03907 0.02015 0.01225 0.008336 0.006133)
			// 10 blue, with 20% noise:
			// numBalls = Discrete(0 0.463 0.2354 0.1137 0.06589 0.04392 0.0322 0.02521 0.02068)
			// 10 blue, with 50% noise:
			// numBalls = Discrete(0 0.125 0.125 0.125 0.125 0.125 0.125 0.125 0.125)
			// 5 blue/5 green, with 20% noise:
			// numBalls = Discrete(0 0.08198 0.09729 0.1102 0.1217 0.1324 0.1425 0.1523 0.1617)
		}
	}
}
