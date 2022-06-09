
#include "nrf_delay.h"
#include "nrf_gpio.h"
#include "nrf.h"


/**
  Single Instruction Multiple Data (SIMD) for simultaneous output of GPIOs 
  Pins to control : 12 pins in total 
*/

#define PULSE_PIN_1 4       // Strain Index Straight
#define PULSE_PIN_2 5       // Strain Index Bent 
#define PULSE_PIN_3 7       // Strain Middle Striaght 
#define PULSE_PIN_4 9       // Strain Middle Bent 
#define PULSE_PIN_5 11      // Strain Ring Straight
#define PULSE_PIN_6 12      // Strain Ring Bent 
#define PULSE_PIN_7 13      // Acc X Positive Swing 
#define PULSE_PIN_8 14      // Acc X Negative Swing 
#define PULSE_PIN_9 15      // Acc Y Positive Swing 
#define PULSE_PIN_10 22     // Acc Y Negative Swing 
#define PULSE_PIN_11 23     // Acc Z Positive Swing 
#define PULSE_PIN_12 24     // Acc Z Negative Swing 


/*
PIN_CNF[n] n = 0 ... 31 = Data Direction Register(DDR)
*/
int main(void)
{
    /*
    pin could be set this way but for simplicity, API call will be used 
    NRF_GPIO->PIN_CNF[LED_GPIO] = (GPIO_PIN_CNF_DIR_Output << GPIO_PIN_CNF_DIR_Pos) |
                                  (GPIO_PIN_CNF_DRIVE_S051 << GPIO_PIN_CNF_DRIVE_Pos) | 
                                  (GPIO_PIN_CNF_INPUT_Connect << GPIO_PIN_CNF_INPUT_Pos) | 
                                  (GPIO_PIN_CNF_PULL_Disabled << GPIO_PIN_CNF_PULL_Pos) | 
                                  (GPIO_PIN_CNF_SENSE_Disabled << GPIO_PIN_CNF_SENSE_Pos); 
    */
    nrf_gpio_cfg_output(PULSE_PIN_1); 
    nrf_gpio_cfg_output(PULSE_PIN_2); 
    nrf_gpio_cfg_output(PULSE_PIN_3); 
    nrf_gpio_cfg_output(PULSE_PIN_4); 
    nrf_gpio_cfg_output(PULSE_PIN_5); 
    nrf_gpio_cfg_output(PULSE_PIN_6); 
    nrf_gpio_cfg_output(PULSE_PIN_7); 
    nrf_gpio_cfg_output(PULSE_PIN_8); 
    nrf_gpio_cfg_output(PULSE_PIN_9); 
    nrf_gpio_cfg_output(PULSE_PIN_10); 
    nrf_gpio_cfg_output(PULSE_PIN_11); 
    nrf_gpio_cfg_output(PULSE_PIN_12); 
   
    // simulate pin 2, 4, 5, 6, 8, 10, 12 to be on
    unsigned long turn_on;
    turn_on |= 1UL << PULSE_PIN_2; 
    turn_on |= 1UL << PULSE_PIN_4;
    turn_on |= 1UL << PULSE_PIN_6; 
    turn_on |= 1UL << PULSE_PIN_8;
    turn_on |= 1UL << PULSE_PIN_10; 
    turn_on |= 1UL << PULSE_PIN_12;
    NRF_GPIO->OUTSET = turn_on;

    // call timer to turn everything off after 2 seconds
    
    while(1)
    {
        
    }    
    
}
/** @} */
