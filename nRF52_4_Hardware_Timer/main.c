

#include <stdbool.h>
#include <stdint.h>
#include "nrf.h"
#include "nrf_drv_timer.h"
#include "bsp.h"
#include "app_error.h"

const nrfx_timer_t TIMER_INSTANCE = NRFX_TIMER_INSTANCE(0);

#define LED1 17
#define LED4 20

void timer0_handler(nrf_timer_event_t event_type, void* p_context) 
{
    switch(event_type)
    {
        case NRF_TIMER_EVENT_COMPARE0:
          nrf_gpio_pin_toggle(LED1); 
          nrf_gpio_pin_toggle(LED4);
          break;
        default:
          break; 
    }
}


void timer_init(void)
{
    uint32_t err_code = NRF_SUCCESS; 
    uint32_t time_ms = 300; 
    uint32_t time_ticks; 

    nrfx_timer_config_t timer_cfg = NRFX_TIMER_DEFAULT_CONFIG; 

    err_code = nrfx_timer_init(&TIMER_INSTANCE, &timer_cfg, timer0_handler); 
    APP_ERROR_CHECK(err_code); 

    // determine after how many ticks the timer interrrupts 
    time_ticks = nrfx_timer_ms_to_ticks(&TIMER_INSTANCE, time_ms);
    
    // assign a channel, pass the number of ticks & enable interrupt 
    nrfx_timer_extended_compare(&TIMER_INSTANCE, NRF_TIMER_CC_CHANNEL0, time_ticks, NRF_TIMER_SHORT_COMPARE0_CLEAR_MASK, true);

    return;
}


int main(void)
{
    nrf_gpio_cfg_output(LED1);
    nrf_gpio_cfg_output(LED4);
    
    nrf_gpio_pin_set(LED1); 
    nrf_gpio_pin_set(LED4); 

    timer_init(); 
    
    nrfx_timer_enable(&TIMER_INSTANCE);


    while (1)
    {
        __WFI();  // turns off clock for the processor : processor in low power mode 
    }
}

/*
Once the timer is initialized, processor clock is turned off 
instruction takes the microcontroller into a low power mode. 
Once the interrupt occurs, the microcontroller gets out of the low power mode 
and timer0_handler is served (LEDs are toggled); 

After exiting from the interrupt, low power mode instruction is called agian
*/
