#include <stdint.h>
#define ADC_BASE_ADDR 	0x4001200UL
#define ADC_CR1_REG_OFFSET 0x04UL

#define ADC_CR1_REG_ADDR (ADC_BASE_ADDR + ADC_CR1_REG_OFFSET)

int main(void)
{
	uint32_t *pADCCr1Reg = (uint32_t*) ADC_CR1_REG_ADDR;

	*pADCCr1Reg |= (1 << 8);

	for(;;);
	return 0;
}
