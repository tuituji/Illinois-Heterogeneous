Illinois-Heterogeneous
======================

Illinois "Heterogeneous Parallel Programming" by Wen-mei W. Hwu

All assignment in skel/*

Linux 
	Edit Makefile according to you own system configuration.
	Then run "make"
	
Windows
	Install Visual Studio, cuda, cmake(select "add to path" when installing)
	Open cmd. Run "cmake CMakeLists.txt".
	Double click hetero.sln and build.
	
!!Notice
	If you are using Visual Studio of version less than 2010, you need a stdint.h
	Copy the "pstdint.h" to the include directory and rename it to "stdint.h"
