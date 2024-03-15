#include <stdbool.h>
#include <stdint.h>

#include "nrf.h"
#include "nordic_common.h"
#include "boards.h"
#include "nrf_delay.h"

#include "nrf_log.h"
#include "nrf_log_ctrl.h"
#include "nrf_log_default_backends.h"
#include "math.h"

void calculateDFT(int* xn)
{
    float Xr[100] = { 0, };
    float Xi[100] = { 0, };
    int i, k, n;
    int N = 0;

    for (k = 0; k < N; k++) {
        Xr[k] = 0;
        Xi[k] = 0;
        for (n = 0; n < 100; n++) {
            Xr[k] = (Xr[k] + *xn[n] * cos(2 * 3.141592 * k * n / N));
            Xi[k] = (Xi[k] - *xn[n] * sin(2 * 3.141592 * k * n / N));
        }
        //TODO: print out each coefficient 
    }
}

/*************************************main***********************************************/ 
int main(void)
{
    APP_ERROR_CHECK(NRF_LOG_INIT(NULL));
    NRF_LOG_DEFAULT_BACKENDS_INIT(); 

    NRF_LOG_INFO("nRF52832 DFT Example"); 
    
    int input_signal[100]; 

    //TODO: signal input logic modification
 
    calculateDFT(input_signal);
    return 0;
}
 