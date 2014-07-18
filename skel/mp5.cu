// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ... + lst[n-1]}

// The <expected_output_file> and <input_file_n> are the input and output files provided in the dataset.
// The <output_file> is the location you¡¯d like to place the output from your program.
// The <type> is the output file type: vector, matrix, or image.
// If an MP does not expect an input or output, then pass none as the parameter.

#include    <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)


__global__ void scan1(float * input, float * output, int len, float* sum0) {
    //@@ Modify the body of this function to complete the functionality of
    //@@ the scan on the device
    //@@ You may need multiple kernel calls; write your kernels before this
    //@@ function and call them from here
	__shared__ float XY[ BLOCK_SIZE * 2 ];

	// to load all the elements 
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(idx * 2 < len){
		XY[threadIdx.x * 2] = input[idx * 2];
	}
	else {
		XY[threadIdx.x * 2] = 0.0;
	}
	if(idx * 2 + 1 < len){
		XY[threadIdx.x * 2 + 1] = input[idx * 2 + 1];
	}
	else {
		XY[threadIdx.x * 2 + 1] = 0.0;
	}
	__syncthreads();

	// 
	for (int stride = 1;stride <= BLOCK_SIZE; stride *= 2) {
		int index = (threadIdx.x + 1) * stride * 2 - 1;
		if (index < 2 * BLOCK_SIZE)
			XY[index] += XY[index - stride];
		__syncthreads();
	}
	
	// 
	for (int stride = BLOCK_SIZE/2; stride > 0; stride /= 2) {
		__syncthreads();
		int index = (threadIdx.x + 1)* stride * 2 - 1;
		if(index + stride < 2 * BLOCK_SIZE) {
			XY[index + stride] += XY[index];
		}
	}
	__syncthreads();
	
	// 
	if (idx * 2 < len) 
		output[idx * 2] = XY[threadIdx.x * 2];
	if (idx * 2 + 1 < len) 
		output[idx * 2 + 1] = XY[threadIdx.x * 2 + 1];
	
	if(threadIdx.x == blockDim.x -1 ) {
		sum0[blockIdx.x] = XY[ BLOCK_SIZE * 2 -1];
	}
}


__global__ void scan(float * input, float * output, int len) {
    //@@ Modify the body of this function to complete the functionality of
    //@@ the scan on the device
    //@@ You may need multiple kernel calls; write your kernels before this
    //@@ function and call them from here
	__shared__ float XY[ BLOCK_SIZE * 2 ];

	// to load all the elements 
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(idx * 2 < len){
		XY[threadIdx.x * 2] = input[idx * 2];
	}
	else {
		XY[threadIdx.x * 2] = 0.0;
	}
	if(idx * 2 + 1 < len){
		XY[threadIdx.x * 2 + 1] = input[idx * 2 + 1];
	}
	else {
		XY[threadIdx.x * 2 + 1] = 0.0;
	}
	__syncthreads();

	// 
	for (int stride = 1;stride <= BLOCK_SIZE; stride *= 2) {
		int index = (threadIdx.x + 1) * stride * 2 - 1;
		if (index < 2 * BLOCK_SIZE)
			XY[index] += XY[index - stride];
		__syncthreads();
	}
	
	// 
	for (int stride = BLOCK_SIZE/2; stride > 0; stride /= 2) {
		__syncthreads();
		int index = (threadIdx.x + 1)* stride * 2 - 1;
		if(index + stride < 2 * BLOCK_SIZE) {
			XY[index + stride] += XY[index];
		}
	}
	__syncthreads();
	
	// 
	if (idx * 2 < len) 
		output[idx * 2] = XY[threadIdx.x * 2];
	if (idx * 2 + 1 < len) 
		output[idx * 2 + 1] = XY[threadIdx.x * 2 + 1];

}

__global__ void scan3(float * input, float * output, int len, float * sum0) {
    //@@ Modify the body of this function to complete the functionality of
    //@@ the scan on the device
    //@@ You may need multiple kernel calls; write your kernels before this
    //@@ function and call them from here

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(blockIdx.x > 0){
		if(idx * 2 < len) output[idx * 2] = input[idx * 2] + sum0[blockIdx.x - 1];
		if(idx * 2 < len + 1) output[idx * 2 + 1] = input[idx * 2 + 1] + sum0[blockIdx.x - 1];
	}
}


int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    float * deviceOutput;
    int numElements; // number of elements in the list

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numElements);
    hostOutput = (float*) malloc(numElements * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ", numElements);

    wbTime_start(GPU, "Allocating GPU memory.");
    wbCheck(cudaMalloc((void**)&deviceInput, numElements*sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceOutput, numElements*sizeof(float)));
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Clearing output memory.");
    wbCheck(cudaMemset(deviceOutput, 0, numElements*sizeof(float)));
    wbTime_stop(GPU, "Clearing output memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    wbCheck(cudaMemcpy(deviceInput, hostInput, numElements*sizeof(float), cudaMemcpyHostToDevice));
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here
	int blocks = (numElements - 1)/(BLOCK_SIZE * 2) + 1;
	int blocks1 = (blocks - 1)/ (BLOCK_SIZE * 2) + 1;
	
	dim3 DimGrid(blocks, 1, 1);
	dim3 DimBlock(BLOCK_SIZE, 1, 1);
	dim3 DimGrid1(blocks1, 1, 1);
	
	wbTime_start(Compute, "Performing CUDA computation");
    //@@ Modify this to complete the functionality of the scan
    //@@ on the deivce
	float * devicesum0;
	wbCheck(cudaMalloc((void**)&devicesum0, blocks * sizeof(float)));
	wbCheck(cudaMemset(devicesum0, 0, blocks * sizeof(float)));
	scan1<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, numElements, devicesum0);
	cudaDeviceSynchronize();
	
	scan<<<DimGrid1, DimBlock>>>(devicesum0, devicesum0, blocks);
	cudaDeviceSynchronize();
	
	scan3<<<DimGrid, DimBlock>>>(deviceOutput, deviceOutput, numElements, devicesum0);
	cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements*sizeof(float), cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, numElements);

    free(hostInput);
    free(hostOutput);

    return 0;
}
