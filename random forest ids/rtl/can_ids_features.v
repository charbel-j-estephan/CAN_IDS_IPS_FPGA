// Streaming CAN feature extractor.
//
// Computes, for every frame, the same ten integer features that features.py
// produces in Python. Deliberately sequential rather than pipelined: one frame
// is handled at a time in a fixed 3 cycles, which removes the read-after-write
// hazard on the per-ID state when two frames of the same ID arrive back to back
// (a diagnostic multiframe transfer does exactly that).
//
// Timing budget: 3 cycles at 100 MHz is 30 ns. The shortest possible CAN frame
// at 1 Mbit/s occupies about 47 us of bus time, so the extractor idles over
// 99.9 % of the time and the 1 ms per frame requirement has three orders of
// magnitude of margin.
//
// Per-ID state lives in RAM indexed by the 11-bit identifier. The learned
// baseline lives in ROM loaded from the .mem files the training flow emits.

`default_nettype none

module can_ids_features #(
    parameter ID_BITS      = 11,
    parameter N_IDS        = 1 << ID_BITS,
    parameter CAP_DT       = 20'hFFFFF,   // 1.048575 s
    parameter CAP_BUS      = 16'hFFFF,    // 65.535 ms
    parameter CAP_RATIO    = 12'hFFF,     // 64x the nominal period
    parameter CAP_BURST    = 4'd15,
    parameter RECIP_SHIFT  = 16,
    parameter MEM_MEAN_INT = "mean_interval.mem",
    parameter MEM_RECIP    = "recip.mem",
    parameter MEM_MEAN_HD  = "mean_hamming.mem",
    parameter MEM_KNOWN    = "id_known.mem",
    parameter MEM_CMASK    = "const_mask.mem",
    parameter MEM_CVAL     = "const_val.mem"
) (
    input  wire                 clk,
    input  wire                 rst_n,

    // one frame in, held stable until busy deasserts
    input  wire                 frame_valid,
    input  wire [ID_BITS-1:0]   can_id,
    input  wire [3:0]           dlc,
    input  wire [63:0]          payload,
    input  wire [31:0]          ts_us,      // free running microsecond counter
    output wire                 busy,

    // features out, valid for one cycle
    output reg                  feat_valid,
    output reg  [19:0]          f_dt_id,
    output reg  [19:0]          f_dt_id_dev,
    output reg  [11:0]          f_dt_ratio_q6,
    output reg  [6:0]           f_hd,
    output reg  [6:0]           f_hd_dev,
    output reg  [15:0]          f_dt_bus,
    output reg  [3:0]           f_burst,
    output reg                  f_id_known,
    output reg  [ID_BITS-1:0]   f_can_id,
    output reg  [3:0]           f_dlc,
    output reg  [6:0]           f_pl_violation,
    output reg  [6:0]           f_pl_popcount
);

    // ------------------------------------------------ helper functions
    function [6:0] popcount64(input [63:0] v);
        integer b;
        reg [6:0] acc;
        begin
            acc = 7'd0;
            for (b = 0; b < 64; b = b + 1)
                acc = acc + {6'd0, v[b]};
            popcount64 = acc;
        end
    endfunction

    // dt / mean_interval in 1/64 steps. One 20x16 multiply by a stored
    // reciprocal, so no divider is needed. 64 means exactly on period.
    function [11:0] ratio_of(input [19:0] dt, input [15:0] rc);
        reg [35:0] prod;
        reg [19:0] shifted;
        begin
            prod     = dt * rc;
            shifted  = prod[RECIP_SHIFT+19:RECIP_SHIFT];
            ratio_of = (shifted > {8'd0, CAP_RATIO}) ? CAP_RATIO
                                                     : shifted[11:0];
        end
    endfunction

    function [19:0] absdiff20(input [19:0] a, input [19:0] b);
        absdiff20 = (a >= b) ? (a - b) : (b - a);
    endfunction

    function [6:0] absdiff7(input [6:0] a, input [6:0] b);
        absdiff7 = (a >= b) ? (a - b) : (b - a);
    endfunction

    // ------------------------------------------------ per-ID state
    reg [31:0] last_ts [0:N_IDS-1];
    reg [63:0] last_pl [0:N_IDS-1];
    reg        seen    [0:N_IDS-1];

    // ------------------------------------------------ learned baseline ROM
    reg [19:0] mean_int [0:N_IDS-1];
    reg [15:0] recip    [0:N_IDS-1];
    reg [6:0]  mean_hd  [0:N_IDS-1];
    reg        id_known [0:N_IDS-1];
    reg [63:0] const_mask [0:N_IDS-1];
    reg [63:0] const_val  [0:N_IDS-1];

    integer k;
    initial begin
        for (k = 0; k < N_IDS; k = k + 1) begin
            last_ts[k] = 32'd0;
            last_pl[k] = 64'd0;
            seen[k]    = 1'b0;
        end
        $readmemh(MEM_MEAN_INT, mean_int);
        $readmemh(MEM_RECIP,    recip);
        $readmemh(MEM_MEAN_HD,  mean_hd);
        $readmemh(MEM_KNOWN,    id_known);
        $readmemh(MEM_CMASK,    const_mask);
        $readmemh(MEM_CVAL,     const_val);
    end

    // ------------------------------------------------ global state
    reg [31:0]        last_bus_ts;
    reg               bus_seen;
    reg [ID_BITS-1:0] prev_id;
    reg               prev_id_valid;
    reg [3:0]         burst_cnt;

    // ------------------------------------------------ control
    localparam S_IDLE = 2'd0, S_READ = 2'd1, S_CALC = 2'd2;
    reg [1:0] state;
    assign busy = (state != S_IDLE);

    // latched frame
    reg [ID_BITS-1:0] r_id;
    reg [3:0]         r_dlc;
    reg [63:0]        r_pl;
    reg [31:0]        r_ts;

    // fetched per-ID state and baseline
    reg [31:0] s_last_ts;
    reg [63:0] s_last_pl;
    reg        s_seen;
    reg [19:0] s_mean_int;
    reg [15:0] s_recip;
    reg [6:0]  s_mean_hd;
    reg        s_known;
    reg [63:0] s_cmask;
    reg [63:0] s_cval;

    // ------------------------------------------------ combinational results
    wire [32:0] dt_id_full  = {1'b0, r_ts} - {1'b0, s_last_ts};
    wire [32:0] dt_bus_full = {1'b0, r_ts} - {1'b0, last_bus_ts};

    wire [19:0] dt_id_sat =
        (!s_seen)                       ? CAP_DT :
        (dt_id_full > {13'd0, CAP_DT})  ? CAP_DT :
                                          dt_id_full[19:0];

    wire [15:0] dt_bus_sat =
        (!bus_seen)                      ? CAP_BUS :
        (dt_bus_full > {17'd0, CAP_BUS}) ? CAP_BUS :
                                           dt_bus_full[15:0];

    wire [6:0] hd_now = popcount64(r_pl ^ s_last_pl);

    // payload bits that break this ID's learned invariant
    wire [6:0] pl_violation_now = popcount64((r_pl ^ s_cval) & s_cmask);
    wire [6:0] pl_popcount_now  = popcount64(r_pl);

    wire       same_as_prev = prev_id_valid && (prev_id == r_id);
    wire [3:0] burst_now    = same_as_prev
                              ? ((burst_cnt == CAP_BURST) ? CAP_BURST
                                                          : burst_cnt + 4'd1)
                              : 4'd0;

    // ------------------------------------------------ sequencer
    always @(posedge clk) begin
        if (!rst_n) begin
            state         <= S_IDLE;
            feat_valid    <= 1'b0;
            last_bus_ts   <= 32'd0;
            bus_seen      <= 1'b0;
            prev_id       <= {ID_BITS{1'b0}};
            prev_id_valid <= 1'b0;
            burst_cnt     <= 4'd0;
        end else begin
            feat_valid <= 1'b0;

            case (state)
            S_IDLE: begin
                if (frame_valid) begin
                    r_id  <= can_id;
                    r_dlc <= dlc;
                    r_pl  <= payload;
                    r_ts  <= ts_us;
                    state <= S_READ;
                end
            end

            S_READ: begin
                s_last_ts  <= last_ts[r_id];
                s_last_pl  <= last_pl[r_id];
                s_seen     <= seen[r_id];
                s_mean_int <= mean_int[r_id];
                s_recip    <= recip[r_id];
                s_mean_hd  <= mean_hd[r_id];
                s_known    <= id_known[r_id];
                s_cmask    <= const_mask[r_id];
                s_cval     <= const_val[r_id];
                state      <= S_CALC;
            end

            S_CALC: begin
                f_dt_id       <= dt_id_sat;
                f_dt_id_dev   <= absdiff20(dt_id_sat, s_mean_int);
                f_dt_ratio_q6 <= ratio_of(dt_id_sat, s_recip);
                f_hd          <= hd_now;
                f_hd_dev      <= absdiff7(hd_now, s_mean_hd);
                f_dt_bus      <= dt_bus_sat;
                f_burst       <= burst_now;
                f_id_known    <= s_known;
                f_can_id      <= r_id;
                f_dlc         <= r_dlc;
                f_pl_violation <= pl_violation_now;
                f_pl_popcount  <= pl_popcount_now;
                feat_valid    <= 1'b1;

                // commit state for the next frame
                last_ts[r_id] <= r_ts;
                last_pl[r_id] <= r_pl;
                seen[r_id]    <= 1'b1;
                last_bus_ts   <= r_ts;
                bus_seen      <= 1'b1;
                prev_id       <= r_id;
                prev_id_valid <= 1'b1;
                burst_cnt     <= burst_now;

                state <= S_IDLE;
            end

            default: state <= S_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
