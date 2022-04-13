#include <stdbool.h>
#include <stdint.h>

#include "nrf.h"
#include "nordic_common.h"
#include "boards.h"
#include "nrf_delay.h"

#include "nrf_log.h"
#include "nrf_log_ctrl.h"
#include "nrf_log_default_backends.h"

/*************************************main***********************************************/ 
int main(void)
{
    APP_ERROR_CHECK(NRF_LOG_INIT(NULL));
    NRF_LOG_DEFAULT_BACKENDS_INIT(); 

    NRF_LOG_INFO("This is log data from nordic device"); 
    uint32_t count = 0;
    
    while(true)
    {
        NRF_LOG_INFO("Counter Value: %d", count); 
        nrf_delay_ms(500); 
        count++; 
    }
}
 