#include "nrf.h"
#include "nordic_common.h"
#include "boards.h"
#include "app_timer.h"

#include "nrf_drv_gpiote.h"
#include "nrf_drv_clock.h"

#include "nrf_gpio.h"
#include "app_error.h"

#define PULSE_PIN_1 4
#define PULSE_PIN_2 5
#define PULSE_PIN_3 7
#define PULSE_PIN_4 11
#define BUTTON_1 13
#define BUTTON_2 14
#define BUTTON_3 15
#define BUTTON_4 16 

#define PULSE_INTERVAL APP_TIMER_TICKS(1000)

APP_TIMER_DEF(m_pulse_timer_id_1); 
APP_TIMER_DEF(m_pulse_timer_id_2); 
APP_TIMER_DEF(m_pulse_timer_id_3); 
APP_TIMER_DEF(m_pulse_timer_id_4); 

/*
This application will turn on and off the GPIO on a 100[ms] interval
GPIOs used : P0.04, P0.05, P0.07, P0.09 

The app timer uses low power timer 32.68kHz
*/


/*************************************timer related***********************************************/ 
static void lfclk_config(void) 
{
    ret_code_t err_code = nrf_drv_clock_init(); 
    nrf_drv_clock_lfclk_request(NULL); 
}

static void app_pulse_1_timer_handler(void * p_context) 
{
    nrf_gpio_pin_clear(PULSE_PIN_1); 
}

static void app_pulse_2_timer_handler(void * p_context) 
{
    nrf_gpio_pin_clear(PULSE_PIN_2); 
}

static void app_pulse_3_timer_handler(void * p_context) 
{
    nrf_gpio_pin_clear(PULSE_PIN_3); 
}

static void app_pulse_4_timer_handler(void * p_context) 
{
    nrf_gpio_pin_clear(PULSE_PIN_4); 
}

static void timers_init(void) 
{
    ret_code_t err_code; 
    err_code = app_timer_init(); 
    APP_ERROR_CHECK(err_code); 

    err_code = app_timer_create(&m_pulse_timer_id_1, APP_TIMER_MODE_SINGLE_SHOT, app_pulse_1_timer_handler); 
    err_code = app_timer_create(&m_pulse_timer_id_2, APP_TIMER_MODE_SINGLE_SHOT, app_pulse_2_timer_handler); 
    err_code = app_timer_create(&m_pulse_timer_id_3, APP_TIMER_MODE_SINGLE_SHOT, app_pulse_3_timer_handler); 
    err_code = app_timer_create(&m_pulse_timer_id_4, APP_TIMER_MODE_SINGLE_SHOT, app_pulse_4_timer_handler); 
} 

/*************************************GPIO Task/Interrupt related***********************************************/ 
static void button_press_handler(nrf_drv_gpiote_pin_t pin, nrf_gpiote_polarity_t action)
{
    if(pin == BUTTON_1)
    {
        nrf_gpio_pin_set(PULSE_PIN_1);
        app_timer_start(m_pulse_timer_id_1, PULSE_INTERVAL, NULL); 
    }
    else if (pin == BUTTON_2)
    {
        nrf_gpio_pin_set(PULSE_PIN_2);
        app_timer_start(m_pulse_timer_id_2, PULSE_INTERVAL, NULL); 
    }
    else if (pin == BUTTON_3) 
    {
        nrf_gpio_pin_set(PULSE_PIN_3);
        app_timer_start(m_pulse_timer_id_3, PULSE_INTERVAL, NULL); 
    }
    else if(pin == BUTTON_4)
    {
        nrf_gpio_pin_set(PULSE_PIN_4);
        app_timer_start(m_pulse_timer_id_4, PULSE_INTERVAL, NULL); 
    }

    
} 

static void gpio_init()
{
    ret_code_t err_code; 
    err_code = nrf_drv_gpiote_init(); 
    
    nrf_drv_gpiote_in_config_t button_config = GPIOTE_CONFIG_IN_SENSE_HITOLO(true); 
    button_config.pull = NRF_GPIO_PIN_PULLUP; 
    
    err_code = nrf_drv_gpiote_in_init(BUTTON_1, &button_config, button_press_handler);
    err_code = nrf_drv_gpiote_in_init(BUTTON_2, &button_config, button_press_handler);
    err_code = nrf_drv_gpiote_in_init(BUTTON_3, &button_config, button_press_handler);
    err_code = nrf_drv_gpiote_in_init(BUTTON_4, &button_config, button_press_handler);
    APP_ERROR_CHECK(err_code); 

    nrf_drv_gpiote_in_event_enable(BUTTON_1, true); 
    nrf_drv_gpiote_in_event_enable(BUTTON_2, true); 
    nrf_drv_gpiote_in_event_enable(BUTTON_3, true); 
    nrf_drv_gpiote_in_event_enable(BUTTON_4, true); 
}

/*************************************main***********************************************/ 
int main(void)
{
    nrf_gpio_cfg_output(PULSE_PIN_1);
    nrf_gpio_cfg_output(PULSE_PIN_2);
    nrf_gpio_cfg_output(PULSE_PIN_3);
    nrf_gpio_cfg_output(PULSE_PIN_4);

    nrf_gpio_pin_clear(PULSE_PIN_1);
    nrf_gpio_pin_clear(PULSE_PIN_2);
    nrf_gpio_pin_clear(PULSE_PIN_3);
    nrf_gpio_pin_clear(PULSE_PIN_4);
     
    lfclk_config(); 
    timers_init(); 
    gpio_init(); 

    

    while(true)
    {
    }
}
 