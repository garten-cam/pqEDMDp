# Introduction #

Welcome to the data-driven system Identification code. In this implementation you will have some tools for (hopefully) modeling, and subsequently analyzing any time-series. The code was developed for modeling dynamical systems, but it should be suitable to identify any time-series.

First things first. This is not a machine learning algorithm, even though, some parts of the solution are suitable for pre-processing data, then, It can be plugged into an ML algorithm.

Now, to the point. All the developments come from the improvement of old, some say classical/traditional, techniques in data-driven system identification. By old, I mean very old, at the core of the algorithm we have the Ordinary Least Squares solution, which is a fancy name for a linear regression.

For this exposition, I will be using the real data from the kiln, and the data pre-processing classes, that where developed to feed "good" data to the algorithm. The "good" data hypothesis is the responsibility of the user, so I will not cover that part in this description/tutorial.

The core of the algorithm is the *extended dynamic mode decomposition* (EDMD). A technique that follows several other types of the decompositions, and that shows a remarkable power to identify dynamical systems in general, regardless of the (possibly) nonlinear behavior of the system to identify or analyze.

Intuitively, the EDMD takes an arbitrary system, linear or nonlinear, and transforms the space of variables into a function space (which is a fancy name for a nonlinear transformation). Once in the function space, it identifies the dynamics of these functions, rather than the dynamics of the states of the system. Why is it convenient to perform this kind of transformation? Because the extra complexity of working with functions is balanced out with the fact that the time-evolution of these functions is linear. Having linearity in the analysis of a system, or an arbitrary time-series is much better than having its nonlinear counterpart. That is why, most of the analysis and control techniques rely on the local linearization of the system.


As stated before, the algorithm is the transformation of the space in which the variables "live" into a function space. Therefore, choosing these functions is an important part of the algorithm, and not selecting them accurately leads to several problems that can make the solution unfeasible. This is the reason why I spent so much time working with orthogonal polynomials and then even more orthogonalizations. With that said, you, as a user, do not have to worry about all the details of that development, the algorithm does it for you. To understand the functionality, let me explain a little bit the architecture of the solution.

pqEDMD is a class that wraps the whole algorithm, it calls the pqObservable class, that provides the "function space" or the set of functions that evaluate and expand/extend the state. Having those functions, the pqEDMD class calls one of the available decompositions and performs the regression.

## Data pre-processing ##

For the data pre-processing, the algorithm used to have its own classes that did a query on a database. This was a very inefficient solution because everybody was designing and implementing their own classes for this purpose. Therefore, this solution does not have dedicated classes for this purpose. Instead, we are using the new data loader-access-cleaning API.
