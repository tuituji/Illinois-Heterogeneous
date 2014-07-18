// The <expected_output_file> and <input_file_n> are the input and output files provided in the dataset.
// The <output_file> is the location you¡¯d like to place the output from your program.
// The <type> is the output file type: vector, matrix, or image.
// If an MP does not expect an input or output, then pass none as the parameter.

// This MP is not correct. Fix it

// Histogram Equalization

#include    <wb.h>

#define HISTOGRAM_LENGTH 256

#define BLOCK_WIDTH 32
#define HIST_LENGTH HISTOGRAM_LENGTH

//@@ insert code here
__global__ void float2uchar(float *input, unsigned char *output, int width, int height, int channels){
	
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	
	if(row < height && col < width){
		for(int i = 0; i < channels; ++i){
			int idx = (row * width + col) * channels + i;
			output[idx] = (unsigned char)(255 * input[idx]);
		}
	}
}

__global__ void rgb2gray(unsigned char *input, unsigned char *output, int width, int height, int channels){

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	
	if(row < height && col < width){
		int idx = (row * width + col);// channels should be 3
		unsigned char r = input[idx * channels];
		unsigned char g = input[idx * channels + 1];
		unsigned char b = input[idx * channels + 2];
		output[idx] = (unsigned char)(0.21*r + 0.71*g + 0.07*b);
	}
}

__global__ void histgramGrayImage(unsigned char *grayImage, unsigned int * histgram, int size){

	__shared__ unsigned int private_histo[HIST_LENGTH];

	if (threadIdx.x < HIST_LENGTH) {
		private_histo[threadIdx.x] = 0;
	}
	__syncthreads();

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	// stride is total number of threads
	int stride = blockDim.x * gridDim.x;
	while (i < size) {
		atomicAdd(&(private_histo[grayImage[i]]), 1);
		i += stride;
	}
	__syncthreads();
	if (threadIdx.x < HIST_LENGTH) {
		atomicAdd( &(histgram[threadIdx.x]), private_histo[threadIdx.x] );
	}
}

#define BLOCK_SIZE  (HIST_LENGTH/2)
__global__ void cdfHistgram(unsigned int *histogram, float *cdf, int size){
	
	__shared__ float XY[ BLOCK_SIZE * 2 ];

	// to load all the elements 
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(idx * 2 < HIST_LENGTH){
		XY[threadIdx.x * 2] = (histogram[idx * 2]) / ((float)size);
	}
	else {
		XY[threadIdx.x * 2] = 0.0;
	}
	if(idx * 2 + 1 < HIST_LENGTH){
		XY[threadIdx.x * 2 + 1] = (histogram[idx * 2 + 1]) / ((float)size);
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
	if (idx * 2 < HIST_LENGTH) 
		cdf[idx * 2] = XY[threadIdx.x * 2];
	if (idx * 2 + 1 < HIST_LENGTH) 
		cdf[idx * 2 + 1] = XY[threadIdx.x * 2 + 1];
}

__global__ void cdfMinmum(float * cdf, float * cdfMin) {
	
	//@@ Load a segment of the input vector into shared memory
	__shared__ float partialMin[2 * BLOCK_SIZE];
	unsigned int t = threadIdx.x;
	unsigned int start = 2 * blockIdx.x * blockDim.x;
	
	if(start + t < HIST_LENGTH ){
		partialMin[t] = cdf[start + t];
	}
	else {
		partialMin[t] = 0.0f;
	}
	if(start + blockDim.x + t < HIST_LENGTH){
		partialMin[blockDim.x + t] = cdf[start + blockDim.x + t];
	}
	else {
		partialMin[blockDim.x + t] = 0.0f;
	}

    //@@ Traverse the reduction tree
	for (unsigned int stride = blockDim.x; stride > 0;  stride /= 2) {
		__syncthreads();
		if (t < stride){
			partialMin[t] = min( partialMin[t], partialMin[t + stride]);
		}
	}
    //@@ Write the computed sum of the block to the output vector at the 
    //@@ correct index
	if(t == 0) {
		cdfMin[0] = partialMin[0];
	}
}

#define clamp(x, val0, val1) (min(max((x), (val0)), (val1)))
__global__ void histogramEqual(unsigned char *input, unsigned char *output, float *cdf, float* cdfmin, 
			int width, int height, int channels){

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	
	if(row < height && col < width){
		for(int i = 0; i < channels; ++i){// ?? 
			int idx = (row * width + col) * channels + i;
			unsigned char val = input[idx];
			//output[idx] = (unsigned char)clamp((int)(255*(cdf[val] - cdfmin[0])/(1 - cdfmin[0])), 0, 255);
			output[idx] = (unsigned char)clamp(((cdf[val] - cdfmin[0])/(1 - cdfmin[0])), 0.0, 255.0);
		}
	}
}

__global__ void rgb2float(unsigned char *input,  float *output, 
			int width, int height, int channels){

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	
	if(row < height && col < width){
		for(int i = 0; i < channels; ++i){// ?? 
			int idx = (row * width + col) * channels + i;
			output[idx] = (float)(input[idx]/ 255.0);
		}
	}
}


int main(int argc, char ** argv) {
    wbArg_t args;
    int imageWidth;
    int imageHeight;
    int imageChannels;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    const char * inputImageFile;

    //@@ Insert more code here
	float *deviceImageDataFloat;
	unsigned char *deviceImageDataUchar;
	unsigned char *deviceImageDataGray;
	unsigned int *deviceHistgram;
	float *deviceCdf;
	float *deviceCdfMin;
	
	float cdf[256];
	unsigned int histgram[256];
	unsigned int total=0;
	float cdfmin;
	
    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);

    wbTime_start(Generic, "Importing data and creating memory on host");
    inputImage = wbImport(inputImageFile);
    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);
    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
    wbTime_stop(Generic, "Importing data and creating memory on host");

	
	wbLog(TRACE, "this is a test");
    //@@ insert code here
	hostInputImageData =wbImage_getData(inputImage);
	hostOutputImageData = wbImage_getData(outputImage);
	cudaMalloc((void **) &deviceImageDataFloat, imageWidth * imageHeight * imageChannels * sizeof(float));
	cudaMalloc((void **) &deviceImageDataUchar, imageWidth * imageHeight * imageChannels * sizeof(unsigned char));
	cudaMalloc((void **) &deviceImageDataGray, imageWidth * imageHeight * sizeof(unsigned char));
	cudaMalloc((void **) &deviceHistgram, HIST_LENGTH * sizeof(unsigned int));
	cudaMalloc((void **) &deviceCdf, HIST_LENGTH * sizeof(float));
	cudaMalloc((void **) &deviceCdfMin , 1 * sizeof(float));
	
    cudaMemcpy(deviceImageDataFloat, hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice);
    
	dim3 dimGrid(ceil((float)imageWidth/BLOCK_WIDTH), ceil((float)imageHeight/BLOCK_WIDTH), 1);
	dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);

	float2uchar<<<dimGrid, dimBlock>>>(deviceImageDataFloat, deviceImageDataUchar, 
								imageWidth, imageHeight, imageChannels);
	
	rgb2gray<<<dimGrid, dimBlock>>>(deviceImageDataUchar, deviceImageDataGray,
								imageWidth, imageHeight, imageChannels);
	
	cudaMemset (deviceHistgram, 0, sizeof(unsigned int)* HIST_LENGTH);
	histgramGrayImage<<<imageHeight, imageWidth>>>(deviceImageDataGray, deviceHistgram, 
								imageWidth * imageHeight);
	
	cudaMemcpy(histgram, deviceHistgram, sizeof(unsigned int) * 256, cudaMemcpyDeviceToHost);
	for(int i = 0; i < 256; ++i){
		wbLog(TRACE, "histgram[", i,"]=" , histgram[i]);
		total += histgram[i];
	}
	wbLog(TRACE, "width = ", imageWidth, "height= ", imageHeight);
	wbLog(TRACE, "channels=", imageChannels);
	wbLog(TRACE, "total=", total, "  w * h=", imageWidth * imageHeight);
	
	//cdfHistgram<<<dim3(1, 1, 1), dim3(BLOCK_SIZE, 1, 1)>>>(deviceHistgram, deviceCdf, imageWidth * imageHeight);
	
	cdfHistgram<<<1, BLOCK_SIZE>>>(deviceHistgram, deviceCdf, imageWidth * imageHeight);
	
	cudaMemcpy(cdf, deviceCdf, sizeof(float) * 256, cudaMemcpyDeviceToHost);
	for(int i = 0; i < 256; ++i)
		wbLog(TRACE, "cdf[", i,"]=" , cdf[i]);
		
	cdfMinmum<<<1, BLOCK_SIZE>>>(deviceCdf, deviceCdfMin);
	cudaMemcpy(&cdfmin,deviceCdfMin, sizeof(float), cudaMemcpyDeviceToHost);
		wbLog(TRACE, "cdfmin",cdfmin);
	
	histogramEqual<<<dimGrid, dimBlock>>>(deviceImageDataUchar, deviceImageDataUchar,
							deviceCdf, deviceCdfMin, imageWidth, imageHeight, imageChannels);
	
	rgb2float<<<dimGrid, dimBlock>>>(deviceImageDataUchar, deviceImageDataFloat, 
							imageWidth, imageHeight, imageChannels);
	
	cudaMemcpy(hostOutputImageData, deviceImageDataFloat,
               imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);
	
	wbSolution(args, outputImage);

    //@@ insert code here

    return 0;
}
