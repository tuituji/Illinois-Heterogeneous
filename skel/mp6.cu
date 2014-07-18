// The <expected_output_file> and <input_file_n> are the input and output files provided in the dataset.
// The <output_file> is the location you��d like to place the output from your program.
// The <type> is the output file type: vector, matrix, or image.
// If an MP does not expect an input or output, then pass none as the parameter.

#include    <wb.h>

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

#define Mask_width  5
#define Mask_radius Mask_width/2

//@@ INSERT CODE HERE

#define TILE_WIDTH 16
#define clamp(x) (min(max((x), 0.0), 1.0))
 
//@@ INSERT CODE HERE
__global__ void convolution(float *I, const float* __restrict__ M, float *P,
                            int channels, int width, int height) {

	__shared__ float Ns[TILE_WIDTH + Mask_width - 1][TILE_WIDTH + Mask_width - 1];
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row_o = blockIdx.y * TILE_WIDTH + ty;
	int col_o = blockIdx.x * TILE_WIDTH + tx;
	int row_i = row_o - Mask_radius;
	int col_i = col_o - Mask_radius;
	
	int i, j, k;
	for (k = 0; k < channels; k++) {
		if((row_i >= 0) && (row_i < height) && (col_i >= 0)  && (col_i < width) ) {
			Ns[ty][tx] = I[(row_i*width + col_i)*channels + k];
		} else{
			Ns[ty][tx] = 0.0f;
		}
		__syncthreads();
		float accum = 0;
		if(tx < TILE_WIDTH && ty < TILE_WIDTH){
			for (j = 0; j < Mask_width; j++){
				for (i = 0; i < Mask_width; i++){
					accum += Ns[threadIdx.y + j][threadIdx.x + i] * M[j * Mask_width + i];
				}
			}
			if (row_o < height && col_o < width){
				P[(row_o * width + col_o) * channels + k] = clamp(accum);
			}
		}
		__syncthreads();
	}
}

int main(int argc, char* argv[]) {
    wbArg_t args;
    int maskRows;
    int maskColumns;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char * inputImageFile;
    char * inputMaskFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    float * hostMaskData;
    float * deviceInputImageData;
    float * deviceOutputImageData;
    float * deviceMaskData;

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);
    inputMaskFile = wbArg_getInputFile(args, 1);

    inputImage = wbImport(inputImageFile);
    hostMaskData = (float *) wbImport(inputMaskFile, &maskRows, &maskColumns);

    assert(maskRows == 5); /* mask height is fixed to 5 in this mp */
    assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);

    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");


    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData,
               hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskData,
               hostMaskData,
               maskRows * maskColumns * sizeof(float),
               cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");


    wbTime_start(Compute, "Doing the computation on the GPU");
    //@@ INSERT CODE HERE
	dim3 dimGrid(ceil((float)imageWidth/TILE_WIDTH), ceil((float)imageHeight/TILE_WIDTH));
    dim3 dimBlock(TILE_WIDTH + Mask_width, TILE_WIDTH + Mask_width, 1);
    convolution<<<dimGrid, dimBlock>>>(deviceInputImageData, deviceMaskData, deviceOutputImageData,
                                       imageChannels, imageWidth, imageHeight);
	
    wbTime_stop(Compute, "Doing the computation on the GPU");


    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData,
               deviceOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    wbSolution(args, outputImage);

    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);

    free(hostMaskData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}
