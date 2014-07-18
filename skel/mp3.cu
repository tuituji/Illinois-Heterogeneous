// Run ./program -e <expected_output_file> -i <input_file_1>,<input_file_2> -o <output_file> -t <type>
// The <expected_output_file> and <input_file_n> are the input and output files provided in the dataset.
// The <output_file> is the location you¡¯d like to place the output from your program.
// The <type> is the output file type: vector, matrix, or image.
// If an MP does not expect an input or output, then pass none as the parameter.

#include    <wb.h>

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)

// Compute C = A * B
__global__ void matrixMultiplyShared(float * A, float * B, float * C,
			             int numARows, int numAColumns,
			             int numBRows, int numBColumns,
			             int numCRows, int numCColumns) {
    //@@ Insert code to implement matrix multiplication here
    //@@ You have to use shared memory for this MP
	__shared__ float sh_a [32][32];
	__shared__ float sh_b [32][32];
	int thdx = threadIdx.x;
	int thdy = threadIdx.y;
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;
	int i, j;
	float value = 0.0;

	for(j = 0; j < (numAColumns-1)/ 32 + 1; ++j){
		if(((j * 32 + thdx) < numAColumns) && (row < numCRows))
			sh_a[thdy][thdx] = A[row * numAColumns + j * 32 + thdx];
		else 
			sh_a[thdy][thdx] = 0.0;
		if(((j * 32 + thdy) < numBRows) && (col < numCColumns))
			sh_b[thdy][thdx] = B[(j * 32 + thdy)* numBColumns + col];
		else 
			sh_b[thdy][thdx] = 0.0;
		__syncthreads();
		for(i = 0; i < 32; ++i) {
			value += sh_a[thdy][i] * sh_b[i][thdx];
		}
		__syncthreads();
	}
	if((row < numCRows) && (col < numCColumns))
		C[row * numCColumns + col] = value;
}

int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostA; // The A matrix
    float * hostB; // The B matrix
    float * hostC; // The output C matrix
    float * deviceA;
    float * deviceB;
    float * deviceC;
    int numARows; // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A
    int numBRows; // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B
    int numCRows; // number of rows in the matrix C (you have to set this)
    int numCColumns; // number of columns in the matrix C (you have to set this)

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostA = (float *) wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
    hostB = (float *) wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
    //@@ Set numCRows and numCColumns
    numCRows = numARows;
    numCColumns = numBColumns;
    //@@ Allocate the hostC matrix
	hostC = (float*)malloc(numCRows * numCColumns * sizeof(float));
	if(hostC == NULL) wbLog(TRACE, "Error allocate memory for hostC");

	wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
    wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
	wbCheck(cudaMalloc((void**)&deviceA, numARows * numAColumns * sizeof(float)));
	wbCheck(cudaMalloc((void**)&deviceB, numBRows * numBColumns * sizeof(float)));
	wbCheck(cudaMalloc((void**)&deviceC, numCRows * numCColumns * sizeof(float)));

    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
	wbCheck(cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice));
	wbCheck(cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice));
	
	
    wbTime_stop(GPU, "Copying input memory to the GPU.");
    
    //@@ Initialize the grid and block dimensions here
    dim3 DimBlock(32, 32, 1);
	dim3 DimGrid((numCColumns-1)/32 + 1, (numCRows-1)/32 + 1, 1);

	wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
	matrixMultiplyShared<<<DimGrid, DimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
	
    cudaThreadSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");
    
    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
	wbCheck(cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(float), cudaMemcpyDeviceToHost));
	
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here
	cudaFree(deviceA);
	cudaFree(deviceB);
	cudaFree(deviceC);
	
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostC, numCRows, numCColumns);

    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}

