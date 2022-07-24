/*
ACK signal and some other signals related to I2C communication is 
handled automatically in the API 
*/ 

#include <stdio.h>
#include "boards.h"
#include "app_util_platform.h"
#include "app_error.h"
#include "nrf_drv_twi.h"

#include "nrf_log.h"
#include "nrf_log_ctrl.h"
#include "nrf_log_default_backends.h"

#define TWI_INSTANCE_ID   0 

static const nrf_drv_twi_t m_twi = NRF_DRV_TWI_INSTANCE(TWI_INSTANCE_ID); 


void twi_init(void)
{
    ret_code_t err_code; 
    const nrf_drv_twi_config_t twi_config = {
        .scl = 22,
        .sda = 23, 
        .frequency = NRF_DRV_TWI_FREQ_100K,
        .interrupt_priority = APP_IRQ_PRIORITY_LOW, 
        .clear_bus_init = false
    }; 


    // can put interrupt handler in the function 
    err_code = nrf_drv_twi_init(&m_twi, &twi_config, NULL, NULL); 
    APP_ERROR_CHECK(err_code); 
    
    nrf_drv_twi_enable(&m_twi); 
}


int main(void)
{
    
}

/** @} */
