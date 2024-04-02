module Sig_ROM #(parameter in_width=10, data_width=16)(
    input                       clk,
    input [in_width-1:0]        x,
    input [data_width-1:0]      out
    );

    reg [data_width-1:0] mem [2**in_width-1:0];
    reg [in_width-1:0] y;

    initial
    begin
        $readmemb("sigContent.mif",mem);
    end

    always @ (posedge clk)
    begin
        if($signed(x) >= 0)
            y <= x+(2**(in_width-1));
        else 
            y <= x-(2**(in_width-1));
    end 


    assign out = mem[y];

endmodule
