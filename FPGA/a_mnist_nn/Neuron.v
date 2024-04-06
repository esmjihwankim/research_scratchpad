`include "include.v"

module Neuron #(parameter layer_number=0, neuron_number=0, num_weight=784, data_width=16, sigmoid_size=5, weight_int_width=1, act_type="relu", bias_file="", weight_file="")(
    input clk, 
    input rst,
    input [data_width-1:0] myinput, 
    input myinput_valid,
    input weight_valid,
    input bias_valid, 
    input [31:0] weight_value, 
    input [31:0] bias_value, 
    input [31:0] config_layer_num,
    input [31:0] config_neuron_num, 
    output [data_width-1:0] out, 
    output reg outvalid
    );
    
    parameter address_width = $clog2(num_weight);

    reg wen;
    wire ren;
    reg [address_width-1:0]     w_addr;
    reg [address_width:0]       r_addr; 
    reg [data_width-1:0]        w_in;
    reg [data_width-1:0]        w_out;
    reg [2*data_width-1:0]      mul;
    reg [2*data_width-1:0]      sum;
    reg [2*data_width-1:0]      bias;
    reg [31:0]                  bias_reg [0:0]
    
    reg                         weight_valid;
    reg                         mult_valid;
    wire                        mux_valid;
    reg                         sig_valid;
    
    wire [2*data_width:0]       combo_add;
    wire [2*data_width:0]       bias_add; 
    reg  [data_width-1:0]       myinputd;
    reg  mux_valid_d;
    reg  mux_valid_f;
    reg  addr=0; 

    always @(posedge clk)
    begin
        if(rst)
        begin
            w_addr <= {address_width{1'b1}};
            wen <= 0; 
        end
        else if(weight_valid & (config_layer_num == layer_number) & (config_neuron_num == neuron_number))
        begin 
            w_in <= weight_value; 
            w_addr <= w_addr + 1; 
            wen <= 1;
        end
        else 
            wen <= 0; 
    end
            
    assign mux_valid     = mult_valid;
    assign combo_add     = mul + sum; 
    assign bias_add      = bias + sum;
    assign ren           = myinput_valid;
    
    `ifdef pretrained 
        initial
        begin
            $readmemb(bias_file, bias_reg);
        end
        always @(posedge clk)
        begin
            bias <= {bias_reg[addr][data_width-1:0], {data_width{1'b0}}}; 
        end
    `else 
        always @(posedge clk)
        begin
            if(bias_valid & (config_layer_num == layer_number) & config_neuron_num == neuron_number)
            begin 
                bias <= {bias_value[data_width-1:0], {data_width{1'b0}}};
            end
        end
    `endif

    always @(posedge clk)
    begin
        if(rst|outvalid)
            r_addr <= 0; 
        else if(myinput_valid)
            r_addr <= r_addr + 1; 
    end

    always @(posedge clk)
    begin
        mul <= $signed(myinputd) *$signed(w_out);
    end



     always @(posedge clk)
     begin
        if(rst|outvalid)
            sum <= 0;
        else if((r_addr == num_weight) & mux_valid_f)
        begin
            // if bias and sum are positive after adding bias to sum,
            // if sign bit becomes 1, saturate 
            if(!bias[2*data_width-1] & !sum[2*data_width-1] & bias_add[2*data_width-1]) 
            begin
                sum[2*data_width-1]    <= 1'b0;
                sum[2*data_width-2:0]  <= {2*data_width-1{1'b1}};
            end
            else if(bias[2*data_width-1] & sum[2*data_width-1] & !bias_add[2*data_width-1])
            begin
                sum[2*data_width-1] <= 1'b1;
                sum[2*data_width-2:0] <= {2*data_width-1{1'b0}};
            end
            else
                sum <= bias_add; 
        end
        else if(mux_valid)
        begin
            // when two positive numbers added and result is negative -> something went wrong : overflow
            if(!mul[2*data_width-1] & !sum[2*data_width-1] & combo_add[2*data_width-1])
            begin
                sum[2*data_width-1]      <= 1'b0;
                sum[2*data_width-2:0]    <= {2*data_width-1{1'b1}};
            end 
            // add two positive numbers and a negative number results -> something went wrong : underflow
            else if (bias[2*data_width-1] & sum[2*data_width-1] * !bias_add[2*data_width-1])
            begin
                sum[2*data_width-1]       <= 1'b1;
                sum[2*data_width-2:0]     <= {2*data_width-1{1'b0}};
            end
            else 
                sum <= bias_add;
        end
        
        else if(mux_valid)
        begin
            if(!mul[2*data_width-1] & !sum[2*data_width-1] & combo_add[2*data_width-1])
            begin
                sum[2*data_width-1]     <= 1'b0;
                sum[2*data_width-2:0]   <= {2*data_width-1{1'b1}};
            end
            else if(mul[2*data_width-1] & sum[2*data_width-1] & !combo_add[2*data_width-1])
            begin
                sum[2*data_width-1] <= 1'b1;
                sum[2*data_width-2:0] <= {2*data_width-1{1'b0}};
            end
            else if(mul[2*data_width-1] & sum[2*data_width-1] & !combo_add[2*data_width-1])
            begin
                sum[2*data_width-1] <= 1'b1;
                sum[2*data_width-2:0] <= {2*data_width-1{1'b0}};
            end
            else 
                sum <= combo_add;
        end
     end

    always @(posedge clk)
    begin
        myinputd        <= myinput;
        weight_valid    <= myinput_valid;
        mult_valid      <= weight_valid;
        sig_valid       <= ((r_addr == num_weight) & mux_valid_f) ? 1'b1 : 1'b0;
        outvalid        <= sig_valid;
        mux_valid_d     <= mux_valid;
        mux_valid_f     <= !mux_valid & mux_valid_d; 
    end 

    // Instantiation of memory for weights
    Weight_Memory #(.num_weight(num_weight), .neuron_number(neuron_number), .layer_number(layer_number), .address_width(address_width), data_width(data_width), .weight_file(weight_file)) WM(
        .clk(clk),
        .wen(wen),
        .ren(ren),
        .wadd(w_addr),
        .radd(r_addr),
        .win(w_in),
        .wout(w_out)
    ); 

    generate
        if(act_type == "sigmoid")
        begin:siginst
            //instantiation of ROM for sigmoid 
            Sig_ROM #(.in_width(sigmoid_size), .data_width(data_width)) s1(
                Sig_ROM #(.in_width(sigmoid_size), .data_width(data_width)) s1(
                    .clk(clk),
                    .x(sum[2*data_width-1:sigmoid_size]),
                    .out(out)
                )
            );
        end
        else 
        begin:ReLUinst
            ReLU #(.data_width(data_width), .weight_int_width(weight_int_width)) s1 (
                .clk(clk),
                .x(sum),
                .out(out)
            ); 
        end 
    endgenerate

    `ifdef DEBUG
    always @(posedge clk)
    begin 
        if(outvalid)
            $display(neuron_number, "%b", out); 
    end
    `endif

endmodule


