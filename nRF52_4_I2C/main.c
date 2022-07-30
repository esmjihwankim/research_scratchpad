
#include <stdio.h>
#include "boards.h"
#include "app_util_platform.h"
#include "app_error.h"
#include "nrf_drv_twi.h"

#include "nrf_log.h"
#include "nrf_log_ctrl.h"
#include "nrf_log_default_backends.h"

#include "ms5611.h"
#include "i2cdev.h"

#include "nrf_delay.h"

#define TWI_INSTANCE_ID 0

static const nrf_drv_twi_t m_twi = NRF_DRV_TWI_INSTANCE(TWI_INSTANCE_ID); 

float pressure;           // Pressure 
float temperature;        // Temperature 
float altitude;           // Altitude value above sea level


int main(void)
{
    ret_code_t err_code; 

    APP_ERROR_CHECK(NRF_LOG_INIT(NULL)); 
    NRF_LOG_DEFAULT_BACKENDS_INIT();
    NRF_LOG_INFO("Application started"); 
    NRF_LOG_FLUSH();
    
    i2cdev_initialize();
    i2cdev_enable(true); 
    
    if(ms5611Init() == true)
    {
        NRF_LOG_INFO("MS5611Init Succeeded"); 
        NRF_LOG_FLUSH();
    }

    while(true) 
    {
        ms5611GetData(&pressure, &temperature, &altitude);
        NRF_LOG_INFO("PRESSURE:: "NRF_LOG_FLOAT_MARKER" ",  NRF_LOG_FLOAT(pressure)); 
        NRF_LOG_FLUSH();
    }

    return 0;
}

/** @} */
