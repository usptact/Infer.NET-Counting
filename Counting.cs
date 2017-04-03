using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;

namespace Counting
{
	class Program
	{
		static void Main(string[] args)
		{
			bool[] data = { true, true, true, true, true, true, true, true, true, true };

			// The probabilistic program
			// -------------------------

			// Variables describing the population
			int maxBalls = 8;
			Range ball = new Range(maxBalls+1); // so that numBalls = (0,...,maxBalls)
			Variable<int> numBalls = Variable.DiscreteUniform(ball);
			VariableArray<bool> isBlue = Variable.Array<bool>(ball);
            // ...add code here...
            isBlue[ball] = Variable.Bernoulli(0.5).ForEach(ball);

            // Variables describing the observations
            Range draw = new Range(data.Length);
			VariableArray<bool> observedBlue = Variable.Array<bool>(draw);
            VariableArray<int> ballIndex = Variable.Array<int>(draw);
            VariableArray<bool> dataArray = Variable.Observed<bool>(data, draw);
            using (Variable.ForEach(draw)) {
                // ...add code here...
                ballIndex[draw] = Variable.DiscreteUniform( numBalls );
                observedBlue[draw] = dataArray[ballIndex[draw]];
			}

            // Inference queries about the program
            // -----------------------------------
            InferenceEngine engine = new InferenceEngine();
            // ...add code here...
            Discrete numberOfBalls = engine.Infer<Discrete>( numBalls );

            Console.WriteLine("Distribution over number of balls: " + numberOfBalls);
			Console.WriteLine("Press any key...");
			Console.ReadKey();

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
