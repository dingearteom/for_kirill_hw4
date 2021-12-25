__kernel void bitonic(__global float *as, unsigned int n, unsigned int k, unsigned int t) {
    unsigned int global_id = get_global_id(0);
    unsigned int local_id = get_local_id(0);
    if (global_id >= n / 2) {
        return;
    }

    __local float local_mem[256];


    unsigned int remainder = global_id % (1 << k);
    unsigned int whole = (global_id / (1 << k)) * (1 << (k + 1));
    unsigned int ind;
    unsigned int dest;
    bool second_part = (remainder >= (1 << (k - 1)));
    unsigned int rr = remainder % (1 << t);
    unsigned int rw = (remainder / (1 << t)) * (1 << (t + 1));
    remainder = rw + rr;
    if (second_part) {
        remainder += (1 << t);
        ind = whole + remainder;
        dest = ind - (1 << t);
    }
    else{
        ind = whole + remainder;
        dest = ind + (1 << t);
    }

    if (as[ind] > as[dest]) {
        float x = as[ind];
        as[ind] = as[dest];
        as[dest] = x;
    }
}
