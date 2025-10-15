#define N 20
#define M 30 

int NUM_OUT_NODE = N;
int NUM_IN_NODE = M;

for (int num_out_node = 0; num_out_node < NUM_OUT_NODE; num_out_node++){
    OT_RSLT[num_out_node] = 0;
    for(int num_in_node = 0; num_in_node < NUM_IN_NODE; num_in_node++){
        OT_RSLT[num_out_node] += IN_NODE[num_in_node] * IN_WEGT[num_out_node][num_in_node]; 
    }
}

