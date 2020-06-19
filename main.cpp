// example for re-producing "gpu_0_res2_0_branch2a_5"

#include <cstdio>
#include <iostream>
#include <cstddef>
#include <fstream>
#include <cstring>

using namespace std;

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

int8_t Dest_cal[1 * 56 * 56 * 64]; // data0012.txt
int8_t Src_cal[1 * 56 * 56 * 64]; // data0013.txt
int8_t Filter_cal[64 * 1 * 1 * 64]; // data0014.txt
int32_t Bias_cal[64]; // data0015.txt

int cnt3; // num of the array
int Dest[1 * 56 * 56 * 64]; // data0012.txt
int Src[1 * 56 * 56 * 64]; // data0013.txt
int Filter[64 * 1 * 1 * 64]; // data0014.txt
int Bias[64]; // data0015.txt
int index_comma[1 * 56 * 56 * 100]; // max length of index_comma
int index_point[1 * 56 * 56 * 100]; // max length of index_point

// read from the graph
uint32_t inWdims[4] = {1, 56, 56, 64};
uint32_t outWdims[4] = {1, 56, 56, 64};
uint32_t filterWdims[4] = {64, 1, 1, 64};
uint32_t biasWdims[1] = {64};
uint32_t kernelSizes[2] = {1, 1};
uint32_t strides[2] = {1, 1};
uint32_t pads[4] = {0, 0, 0, 0};


int8_t libjit_clip(int32_t val) {
  return (int8_t)MIN(MAX(val, -128), 127);
}

int32_t libjit_scale_i32i8(int32_t input, int32_t pre, int32_t post,
                                  int32_t scale, int32_t offset) {
	int rtn = (post > 0) ? (1 << (post - 1)) : 0;
    return ((((input >> pre) * scale) + rtn) >> post) + offset;
}

uint32_t libjit_getXYZW(uint32_t *dims, int32_t x, int32_t y, int32_t z,
                            int32_t w) {
  return (x * dims[1] * dims[2] * dims[3]) + (y * dims[2] * dims[3]) +
         (z * dims[3]) + w;
}


void StringToNum(string s, int * pdata)
{
    int cnt1 = 0, cnt2 = 0; // cnt1 for comma, cnt2 for point
    for (int i = 0;i < s.length();i++) {
        if (s[i] == ',') {
            index_comma[cnt1] = i;
            cnt1++;
        } else if (s[i] == '.') {
            index_point[cnt2] = i;
            cnt2++;
        }
    }

    cnt3 = 0; // cnt3 for pdata
    int data = 0;
    int first_index = 0;
    if (s[0]>='0' && s[0] <= '9') {
        data *= 10;
        data += int(s[first_index] - '0');
        first_index++;
        while(first_index != index_point[0]) {
            data *= 10;
            data += int(s[first_index] - '0');
            first_index++;
        }
        pdata[cnt3] = data;
        cnt3++;
    } else {
        first_index = 1;
        data += int(s[first_index] - '0');
        first_index++;
        while(first_index != index_point[0]) {
            data *= 10;
            data += int(s[first_index] - '0');
            first_index++;
        }
        pdata[cnt3] = (-1) * data;
        cnt3++;
    }

    // cout << "cnt1 = " << cnt1 << endl;
    for (int i = 0;i < cnt1 - 1;i++) {
        data = 0;
        // cout << "hello " << endl;
        int j = index_comma[i] + 2;
            // cout << "j = " << j << endl;
            if (s[j] >= '0' && s[j] <= '9') {
                while (j != index_point[i + 1]) {
                    data *= 10;
                    data += int(s[j] - '0');
                    j++;
                    // cout << data << endl;
                }
                pdata[cnt3] = data;
                // cout << Src[cnt3] << endl;
                cnt3++;
            } else if (s[j] == '-') {
                j++;
                while (j != index_point[i + 1]) {
                    data *= 10;
                    data += int(s[j] - '0');
                    j++;
                    // cout << data << endl;
                }
                pdata[cnt3] = (-1) * data;
                // cout << Src[cnt3] << endl;
                cnt3++;
            }   
    }
}

// gpu_0_res2_0_branch2a_5
/*

destOffset = -128
srcOffset = -128
filterOffset = 22
biasOffset = 0
biasPre = 0
biasPost = 0
biasScale = 1
outPre = 1
outPost = 15
outScale = 420
*/


void quantized_conv2d(int8_t *outW, int8_t *inW, int8_t *filterW,
    int32_t *biasW, uint32_t *outWdims, uint32_t *inWdims,
    uint32_t *filterWdims, uint32_t *biasWdims, uint32_t *kernelSizes,
    uint32_t *strides, uint32_t *pads, int32_t group, int32_t outOffset,
    int32_t inOffset, int32_t filterOffset, int32_t biasOffset, int32_t biasPre,
    int32_t biasPost, int32_t biasScale, int32_t outPre, int32_t outPost,
    int32_t outScale, unsigned depthUnroll, uint32_t dilation) {

	int32_t inChannels = inWdims[3];
  	int32_t outChannels = outWdims[3];
  	int32_t inCperG = inChannels / group;
  	int32_t outCperG = outChannels / group;
  	int32_t pad_t = pads[0];
  	int32_t pad_l = pads[1];
  	int32_t stride_h = strides[0];

  	size_t stride_w = strides[1];
  	size_t kernel_h = kernelSizes[0];
  	size_t kernel_w = kernelSizes[1];


  	for (size_t n = 0; n < inWdims[0]; n++) {
    	for (size_t g = 0; g < group; g++) {
      		for (size_t d = g * outCperG; d < (g + 1) * outCperG; d += depthUnroll) {
        			ssize_t x = -(ssize_t)pad_t;
        		for (size_t ax = 0; ax < outWdims[1]; x += stride_h, ax++) {
          				ssize_t y = -(ssize_t)pad_l;
          			for (size_t ay = 0; ay < outWdims[2]; y += stride_w, ay++) {
            				int32_t sum[depthUnroll];
						for (unsigned i = 0; i < depthUnroll; i++) {
              					sum[i] = libjit_scale_i32i8((int32_t)biasW[d + i] - biasOffset,
                                      		biasPre, biasPost, biasScale, 0);
            					}

            				for (size_t fx = 0; fx < kernel_h; fx++) {
              					for (size_t fy = 0; fy < kernel_w; fy++) {
                						ssize_t ox = x + fx * dilation, oy = y + fy * dilation;

                					if (ox < 0 || oy < 0 || ox >= (ssize_t)inWdims[1] || oy >= (ssize_t)inWdims[2]) {
                  									continue;
        								}

                					size_t inIdx = libjit_getXYZW(inWdims, n, (size_t)ox,
                                              (size_t)oy, g * inCperG);
                size_t filterIdx = libjit_getXYZW(filterWdims, d, fx, fy, 0);
                size_t sliceSize = filterWdims[1] * filterWdims[2] * filterWdims[3];

                for (size_t fd = 0; fd < inCperG; fd++) {
                  int32_t in = inW[inIdx + fd] - inOffset;
                  for (unsigned i = 0; i < MIN(4, depthUnroll); i++) {
                    sum[i] += (filterW[filterIdx + (sliceSize * i) + fd] -
                               filterOffset) *
                              in;
                  }
                }

                // if (depthUnroll > 4)
                //   for (size_t fd = 0; fd < inCperG; fd++) {
                //     int32_t in = inW[inIdx + fd] - inOffset;
                //     for (unsigned i = 4; i < MIN(8, depthUnroll); i++) {
                //       sum[i] += (filterW[filterIdx + (sliceSize * i) + fd] -
                //                  filterOffset) *
                //                 in;
                //     }
                //   }
              }
            }

            for (unsigned i = 0; i < depthUnroll; i++) {
              int32_t scaledSum = libjit_scale_i32i8(sum[i], outPre, outPost,
                                                     outScale, outOffset);
              outW[libjit_getXYZW(outWdims, n, ax, ay, d + i)] =
                  libjit_clip(scaledSum);
            }
          } // W
        }   // H
      }     // C
    }       // G
  }         // N
}


int main(void)
{
    ifstream infile_src;
    infile_src.open("./data/data0013.txt");
    memset(index_point, 0, sizeof(index_point));
    memset(index_comma, 0, sizeof(index_comma));
    // int8_t * ptr = &array[0];
    int flag = 0;
    string src;
    // ignore the first line
    while(getline(infile_src, src)) {
        if (!flag) {
            flag++;
            continue;
        }
    }
    infile_src.close();
    StringToNum(src, Src);
    cout << cnt3 << endl;
    for (int i = 0;i < cnt3;i++) {
    	Src_cal[i] = (int8_t)Src[i];
    }

    ifstream infile_filter;
    infile_filter.open("./data/data0014.txt");
    memset(index_point, 0, sizeof(index_point));
    memset(index_comma, 0, sizeof(index_comma));
    flag = 0;
    string filter;
    while(getline(infile_filter, filter)) {
    	if (!flag) {
    		flag++;
    		continue;
    	}
    }
    infile_filter.close();
    StringToNum(filter, Filter);
    cout << cnt3 << endl;
    for (int i = 0;i < cnt3;i++) {
    	Filter_cal[i] = (int8_t)Filter[i];
    }


    ifstream infile_bias;
    infile_bias.open("./data/data0015.txt");
    memset(index_point, 0, sizeof(index_point));
    memset(index_comma, 0, sizeof(index_comma));
    flag = 0;
    string bias;
    while(getline(infile_bias, bias)) {
    	if (!flag) {
    		flag++;
    		continue;
    	}
    }
    infile_bias.close();
    StringToNum(bias, Bias);
    cout << cnt3 << endl;
    for (int i = 0;i < cnt3;i++) {
    	Bias_cal[i] = (int32_t)Bias[i];
    }

    ifstream infile_dest;
    infile_dest.open("./data/data0012.txt");
    memset(index_point, 0, sizeof(index_point));
    memset(index_comma, 0, sizeof(index_comma));
    flag = 0;
    string dest;
    while(getline(infile_dest, dest)) {
    	if (!flag) {
    		flag++;
    		continue;
    	}
    }
    infile_dest.close();
    StringToNum(dest, Dest);
    cout << cnt3 << endl;


    // test for quantized_conv2d
	quantized_conv2d(Dest_cal, Src_cal, Filter_cal, 
		Bias_cal, outWdims, inWdims, 
		filterWdims, biasWdims, kernelSizes,
		strides, pads, 1, -128, -128, 22, 0, 0, 0, 1, 1, 15, 
		420, 1, 1);

	int correct = 0;
	for (int i = 0;i < 1 * 56 * 56 * 64;i++) {
		// Print all of the information about Dest_cal and Dest
		cout << "Calculate: " << (int32_t)Dest_cal[i] << " " << "Extracted: "<< Dest[i]<< endl;
		if ((int32_t)Dest_cal[i] == Dest[i]) {
			correct++;
		}
	}
	int all_num = 1 * 56 * 56 * 64;
	if (correct == all_num) {
		cout << "Pass!" << endl;
		cout << "{" << correct << "/" << all_num << "}" << endl;
	} else {
		cout << "Wrong!" << endl;
		cout << "{" << correct << "/" << all_num << "}" << endl;
	}



	return 0;
}