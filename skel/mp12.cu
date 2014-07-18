// The <expected_output_file> and <input_file_n> are the input and output files provided in the dataset.
// The <output_file> is the location you¡¯d like to place the output from your program.
// The <type> is the output file type: vector, matrix, or image.
// If an MP does not expect an input or output, then pass none as the parameter.

#include	<wb.h>

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)

#define SegSize   (1024 * 1024)

__global__ void vecAdd(float * in1, float * in2, float * out, int len) {
    //@@ Insert code to implement vector addition here
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < len) {
		out[tid] = in1[tid] + in2[tid];
	}
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

	cudaStream_t stream0, stream1;
	wbCheck(cudaStreamCreate(&stream0));
	wbCheck(cudaStreamCreate(&stream1));
	
	float *d_A0, *d_B0, *d_C0;// device memory for stream 0
	float *d_A1, *d_B1, *d_C1;// device memory for stream 1

	wbCheck(cudaMalloc((void**)&d_A0, SegSize * sizeof(float)));
    wbCheck(cudaMalloc((void**)&d_B0, SegSize * sizeof(float)));
    wbCheck(cudaMalloc((void**)&d_C0, SegSize * sizeof(float)));
	
	wbCheck(cudaMalloc((void**)&d_A1, SegSize * sizeof(float)));
    wbCheck(cudaMalloc((void**)&d_B1, SegSize * sizeof(float)));
    wbCheck(cudaMalloc((void**)&d_C1, SegSize * sizeof(float)));
	
	wbTime_start(Generic, "Start Ansy kernel");
	for (int i=0; i < (inputLength - 1)/(SegSize * 2) + 1; i += SegSize * 2) {
		//int left = inputLength - i;
		//if(left)
		// ignore if the last seg is less than the SegSize;
		cudaMemcpyAsync(d_A0, hostInput1 + i, SegSize*sizeof(float), cudaMemcpyHostToDevice, stream0);
		cudaMemcpyAsync(d_B0, hostInput2 + i, SegSize*sizeof(float), cudaMemcpyHostToDevice, stream0);
		
		cudaMemcpyAsync(d_A1, hostInput1 + i + SegSize, SegSize*sizeof(float), cudaMemcpyHostToDevice, stream0);
		cudaMemcpyAsync(d_B1, hostInput2 + i + SegSize, SegSize*sizeof(float), cudaMemcpyHostToDevice, stream0);
		
		vecAdd<<<SegSize/256, 256, 0, stream0>>>(d_A0, d_B0, d_C0, SegSize);
		
		vecAdd<<<SegSize/256, 256, 0, stream1>>>(d_A1, d_B1, d_C1, SegSize);
		
		cudaMemcpyAsync(hostOutput + i, d_C0, SegSize * sizeof(float), cudaMemcpyDeviceToHost, stream0);
		cudaMemcpyAsync(hostOutput + i + SegSize, d_C0, SegSize * sizeof(float), cudaMemcpyDeviceToHost, stream0);
	}
	wbTime_stop(Generic, "Finish Ansy kernel");
	
    wbSolution(args, hostOutput, inputLength);

	cudaFree(d_A0);
	cudaFree(d_B0);
	cudaFree(d_C0);
	cudaFree(d_A1);
	cudaFree(d_B1);
	cudaFree(d_C1);
    free(hostInput1);
    free(hostInput2);
    free(hostOutput);

    return 0;
}
