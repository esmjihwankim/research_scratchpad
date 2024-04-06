module ReLU #(parameter data_width=16, weight_int_width=4) (
    input                       clk,
    input [2*data_width-1:0]    x,
    output reg [data_width-1:0] out
)

always @(posedge clk)
begin
    if($signed(x) >= 0)
    begin
        if(|x[2*data_width-1-:weight_int_width+1])
            out <= {1'b0,{(data_width-1){1'b1}}};
        else 
            out <= x[2*data_width-1-weight_int_width-:data_width];
    end
    else 
        out <= 0;
end

endmodule