

#include <stdbool.h>
#include <stdint.h>
#include "nrf.h"
#include "nrf_drv_timer.h"
#include "bsp.h"
#include "app_error.h"



/**
 * @brief Function for main application entry.
 */
int main(void)
{


    while (1)
    {
        __WFI();
    }
}

/** @} */
