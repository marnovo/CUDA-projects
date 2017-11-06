#include <iostream>
#include <math.h>
#include <string>

//****************************************************************************80
//
//    ADD adds elements of two arrays on CPU
//

void add_arrays(int n, float *x, float *y)
{
    for (int i = 0; i < n; i++)
    y[i] = x[i] + y[i];
}

//****************************************************************************80
//
//    MAIN is the main program for array element addition on CPU example.
//

int main(void)
{
    int N = pow(10, 9); // one billion elements

    float *x = new float[N];
    float *y = new float[N];

    std::string result = "";

    // initialize x and y arrays
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // add elements (on CPU)
    add_arrays(N, x, y);
  
    // free mem when done
    delete [] x;
    delete [] y;

    // print array length
    result = "length of array = " + std::to_string(N);
    std::cout << result << "\n";

    return 0;
}
