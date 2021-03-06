// MP 1

// Run ./program -e <expected_output_file> -i <input_file_1>,<input_file_2> -o <output_file> -t <type>
// The <expected_output_file> and <input_file_n> are the input and output files provided in the dataset. 
// The <output_file> is the location you��d like to place the output from your program. 
// The <type> is the output file type: vector, matrix, or image. 
// If an MP does not expect an input or output, then pass none as the parameter.


#include	<wb.h>

__global__ void vecAdd(float * in1, float * in2, float * out, int len) {
    //@@ Insert code to implement vector addition here
	int i = threadIdx.x+blockDim.x*blockIdx.x;
	if(i < len) out[i] = in1[i] + in2[i];
}

int main(int argc, char ** argv) {
    wbArg_t args;
    int inputLength;
    float * hostInput1;
    float * hostInput2;
    float * hostOutput;
    float * deviceInput1;
    float * deviceInput2;
    float * deviceOutput;

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput1 = (float *) wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostInput2 = (float *) wbImport(wbArg_getInputFile(args, 1), &inputLength);
    hostOutput = (float *) malloc(inputLength * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The input length is ", inputLength);

	wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
	cudaError_t err;
	err = cudaMalloc((void **) &deviceInput1, inputLength * sizeof(float));
	if(err != cudaSuccess) wbLog(TRACE, "error allocate mem for deviceInput1"); 
	err = cudaMalloc((void **) &deviceInput2, inputLength * sizeof(float));
	if(err != cudaSuccess) wbLog(TRACE, "error allocate mem for deviceInput2"); 
	err = cudaMalloc((void **) &deviceOutput, inputLength * sizeof(float));
	if(err != cudaSuccess) wbLog(TRACE, "error allocate mem for deviceOutput"); 
	
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
	err = cudaMemcpy(deviceInput1, hostInput1, inputLength * sizeof(float), cudaMemcpyHostToDevice);
	if(err != cudaSuccess) wbLog(TRACE, "error copy hostInput1 to gpu"); 
	err = cudaMemcpy(deviceInput2, hostInput2, inputLength * sizeof(float), cudaMemcpyHostToDevice);
	if(err != cudaSuccess) wbLog(TRACE, "error copy hostInput2 to gpu"); 
	
    wbTime_stop(GPU, "Copying input memory to the GPU.");
    
    //@@ Initialize the grid and block dimensions here
	dim3 DimBlock(1024, 1, 1);
	dim3 DimGrid((inputLength-1)/1024 + 1, 1, 1);
    
    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
	vecAdd<<<DimGrid, DimBlock>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);

	cudaThreadSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");
    
    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
	err = cudaMemcpy(hostOutput, deviceOutput, inputLength * sizeof(float), cudaMemcpyDeviceToHost);
	
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here
	cudaFree(deviceOutput);
	cudaFree(deviceInput1);
	cudaFree(deviceInput2);

    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, inputLength);

    free(hostInput1);
    free(hostInput2);
    free(hostOutput);

    return 0;
}
