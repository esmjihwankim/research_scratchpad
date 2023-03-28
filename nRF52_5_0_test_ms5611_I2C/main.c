
#include <stdio.h>
#include "boards.h"
#include "app_util_platform.h"
#include "app_error.h"
#include "nrf_drv_twi.h"

#include "nrf_log.h"
#include "nrf_log_ctrl.h"
#include "nrf_log_default_backends.h"

// hardware timer : for systick 
#include "nrf_drv_timer.h"

// i2c and i2c sensor related
#include "ms5611.h"
#include "i2cdev.h"

#define TWI_INSTANCE_ID 0

static const nrf_drv_twi_t m_twi = NRF_DRV_TWI_INSTANCE(TWI_INSTANCE_ID); 

// value to be updated from the hardware timer 
volatile int ms_ticks = 0; 

// values to be calculated from ms5611 sensor  
float pressure;         
float temperature;       
float altitude;           

const nrfx_timer_t HW_TIMER_INSTANCE = NRFX_TIMER_INSTANCE(0); 

void timer0_handler(nrf_timer_event_t event_type, void* p_context)
{
    switch(event_type)
    {
        case NRF_TIMER_EVENT_COMPARE0:
          ms_ticks += 1; 
          break;
        default:
          break;
    }
}

void hw_timer_init(void)
{
    uint32_t err_code;
    uint32_t time_ms = 1; 
    uint32_t ticks; 

    nrfx_timer_config_t timer_cfg = NRFX_TIMER_DEFAULT_CONFIG;

    err_code = nrfx_timer_init(&HW_TIMER_INSTANCE, &timer_cfg, timer0_handler); 
    APP_ERROR_CHECK(err_code);

    ticks = nrfx_timer_ms_to_ticks(&HW_TIMER_INSTANCE, time_ms);

    // last parameter : enable interrupt
    nrfx_timer_extended_compare(&HW_TIMER_INSTANCE, NRF_TIMER_CC_CHANNEL0, ticks, NRF_TIMER_SHORT_COMPARE0_CLEAR_MASK, true); 

    return; 
}


int main(void)
{
    ret_code_t err_code; 

    APP_ERROR_CHECK(NRF_LOG_INIT(NULL)); 
    NRF_LOG_DEFAULT_BACKENDS_INIT();
    NRF_LOG_INFO("Application started"); 
    NRF_LOG_FLUSH();
    
    // 1ms hardware timer 
    hw_timer_init();
    nrfx_timer_enable(&HW_TIMER_INSTANCE); 
    
    // i2c 
    i2cdev_initialize();
    i2cdev_enable(true); 
    
    if(ms5611Init() == true)
    {
        NRF_LOG_INFO("MS5611Init Succeeded"); 
        NRF_LOG_FLUSH();
    }


    while(true) 
    {
        // raw temperature 
        temperature = ms5611RawTemperature(MS5611_OSR_4096);
        NRF_LOG_INFO("Temperature:: "NRF_LOG_FLOAT_MARKER" ",   NRF_LOG_FLOAT(temperature));
        //NRF_LOG_INFO("Pressure :: "NRF_LOG_FLOAT_MARKER" ",     NRF_LOG_FLOAT(pressure));      
        NRF_LOG_FLUSH();
    }

    return 0;
}

/** @} */
