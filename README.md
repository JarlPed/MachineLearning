# MachineLearning
Machine learning examples, using python.

The motivation for this repository is to provide machine learning examples and compare its performance.

Prerequisite
`pip install -r requirements.txt`

Examples
1: A simple PID minimization of the OpenAI PoleCart i.e. inverted pendulum is solved with a simple Newton-Raphson maximizer using central gradients. No Neural networks is used, but it's useful to compare other AI structures. Run the example by: `python simplePIDcontoller.py`
This example is constructed to sucseed, however Newtons method is known for unstability if parameters have large deviations from the maxima/minima. To further implicate, the inverse pendulum simulation is not determinstic, therefore bias is introduced for each trial (except when the problem is solved). Changing the parameters to unity, the method is very likely to diverge. Note that only 2 operations is allowed for the simulation, 1/0; 1 = rightward push, 0 = leftward push. Outcomes could be different if a continual set of operations were allowed, i.e. real numbers in [-1,1].
