Illinois-Heterogeneous
======================

Illinois "Heterogeneous Parallel Programming" by Wen-mei W. Hwu
http://webgpu.hwu.crhc.illinois.edu/
https://www.coursera.org/course/hetero


All assignment in skel/*

Linux :
1.Edit Makefile according to you own system configuration.
2.Then run "make"
	
Windows :
1.Install Visual Studio, cuda, cmake(select "add to path" when installing)
2.Open cmd. Run "cmake CMakeLists.txt".
3.Double click hetero.sln and build.
	
Notice!!
If you are using Visual Studio of version less than 2010, you need a stdint.h. Copy the "pstdint.h" to the include directory and rename it to "stdint.h"
