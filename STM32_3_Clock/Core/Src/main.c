#include <stdint.h>
#define ADC_BASE_ADDR 			0x4001200UL
#define ADC_CR1_REG_OFFSET 		0x04UL
#define ADC_CR1_REG_ADDR 		(ADC_BASE_ADDR + ADC_CR1_REG_OFFSET)

#define RCC_BASE_ADDR 			0x40023800UL
#define RCC_APB2_ENR_OFFSET 	0x44UL
#define RCC_APB2_ENR_ADDR 		(RCC_BASE_ADDR + RCC_APB2_ENR_OFFSET)


int main(void)
{
	uint32_t *pADCCr1Reg = (uint32_t*) ADC_CR1_REG_ADDR;
	uint32_t *pRccApb2Enr = (uint32_t*) RCC_APB2_ENR_ADDR;

	// 1. Enable the peripheral clock for ADC1
	*pRccApb2Enr |= (1 << 8);

	// 2. Modify the ADC cr1 register
	*pADCCr1Reg |= (1 << 8);

	for(;;);
	return 0;
}
