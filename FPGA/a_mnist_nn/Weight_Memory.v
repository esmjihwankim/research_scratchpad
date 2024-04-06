`include "include.v"

module Weight_Memory # (parameter numWeight=3, neuron_number=5, layer_number=1, address_width=10, data_width=16, weight_file="w_1_15.mif"
    input clk, 
    input wen,
    input ren, 
    input [address_width-1:0] wadd,
    input [address_width-1:0] radd, 
    input [data_width-1:0] win,
    output reg [data_width-1:0] wout);

    reg [data_width-1:0] mem [number_weight-1:0]; 
    
    // if defined, it will act like ROM
    // if not defined, it will act like RAM
    `ifdef pretrained
        initial 
        begin
            $readmemb(weight_file, mem);
        end
    `else 
        always @(posedge clk) 
        begin
            if (wen) 
            begin
                mem[wadd] <= win;
            end
        end 
    `endif 

    always @ (posedge clk)
    begin
        
    
endmodule