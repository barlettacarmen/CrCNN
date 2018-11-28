# Introduction
SEAL (Simple Encrypted Arithmetic Library) is an easy-to-use homomorphic encryption 
library, developed by researchers in the Cryptography Research group at Microsoft 
Research. SEAL is written in standard C++17 and contains .NET Framework wrappers 
for the public API. SEAL is licensed under the Microsoft Research License Agreement 
and is free for research use (see LICENSE.txt).

# System requirements
Since SEAL has no external dependencies and is written in standard C++17 it is 
easy to build on any 64-bit system. For building in Windows, SEAL contains a Visual 
Studio 2017 solution file. For building in Linux and Mac OS X, SEAL requires either 
g++-6 or newer, or clang++-5 or newer. Please see INSTALL.txt for installation 
instructions using CMake.

# Documentation
The code-base contains examples written both in C++ and in C#. These examples are 
heavily commented, serving as a self-contained short introduction to using SEAL.
In addition, the .h files both in the C++ library and in the .NET wrapper library 
contain detailed comments for the public API.

To learn to use SEAL, we recommend looking at the example projects, as they 
contain several detailed and thoroughly commented examples demonstrating most of 
the basic features.

# Acknowledgements
We would like to thank John Wernsing, Michael Naehrig, Nathan Dowlin, Rachel 
Player, Gizem Cetin, Susan Xia, Peter Rindal, Kyoohyung Han, Zhicong Huang, and 
Amir Jalali for their contributions to the SEAL project. We would also like to 
thank everyone who has sent us helpful comments, suggestions, and bug reports.

# Contact Us
If you have any questions, suggestions, comments, or bug reports, please contact 
us by emailing [sealcrypto@microsoft.com](mailto:sealcrypto@microsoft.com).
