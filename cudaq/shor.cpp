#include <iostream>
#include <cstdlib>
#include <cudaq.h>
#include <numeric>
#include <numbers>
#include <span>
#include <cmath>
#include <chrono>

// Helper functions
std::vector<int> num2bin(long long value, int n_bits);
std::vector<double> compute_phi_add_angles(long long a, int n);
std::pair<long long, long long> extended_gcd(long long a, long long b);
long long mod_inverse(long long a, long long N);

// Draper QFT adder with & w/o control qubits
void __qpu__ phi_add(cudaq::qview<> qubits, const std::vector<double>& angles, int start); 
void __qpu__ phi_add(cudaq::qview<> qubits, cudaq::qubit& ctrl, const std::vector<double>& angles, int start); 
// Inverse of phi_add
void __qpu__ phi_sub(cudaq::qview<> qubits, const std::vector<double>& angles, int start);
void __qpu__ phi_sub(cudaq::qview<> qubits, cudaq::qubit& ctrl, const std::vector<double>& angles, int start);

// Controlled multiplier gate
void __qpu__ CMULT(
    cudaq::qview<> xreg,
    cudaq::qview<> qubits, 
    cudaq::qubit& ancilla, 
    const std::vector<double>& a_angles, 
    const std::vector<double>& N_angles, 
    int angle_start);
void __qpu__ iCMULT(
    cudaq::qview<> xreg, 
    cudaq::qview<> qubits, 
    cudaq::qubit& ancilla, 
    const std::vector<double>& a_angles, 
    const std::vector<double>& N_angles, 
    int angle_start);

// Modular exponentiation
void __qpu__ Ua(
    cudaq::qview<> xreg, 
    cudaq::qview<> qubits, 
    cudaq::qubit& ancilla, 
    const std::vector<double>& a_angles, 
    const std::vector<double>& a_inv_angles, 
    const std::vector<double>& N_angles, 
    int angle_start);

// Shor's algorithm kernel
void __qpu__ Shor(
    long long N, 
    long long a, 
    int t, 
    int n, 
    const std::vector<double>& Ua_angles, 
    const std::vector<double>& Ua_inv_angles, 
    const std::vector<double>& N_angles, 
    const std::vector<int>& angle_starts); 

// Continued fraction convergents
std::set<long long> continued_fractions(double phase, long long Q, long long max_den);
// Post-processing function
void post_process_shor(
    const cudaq::sample_result& result,   
    long long N,
    long long a,
    int t,
    int min_shots_threshold          
);


struct qft {
    void operator()(cudaq::qview<> qubits) __qpu__ {
        int n = qubits.size();

        // qubits[0] is LSB, we iterate from n-1 down to 0
        for (int i = n - 1; i >= 0; i--) {
            h(qubits[i]);
            
            for (int j = i - 1; j >= 0; j--) {
                double angle = M_PI / (1LL << (i-j));  // 1 << (i-j) = 2^(i-j) 
                r1<cudaq::ctrl>(angle, qubits[j], qubits[i]);            
            }
        }
        
        // swap 
        for (int i = 0; i < n / 2; i++) {
            swap(qubits[i], qubits[n - i - 1]);
        }
    }
};


struct iqft {
    void operator()(cudaq::qview<> qubits) __qpu__ {
        int n = qubits.size();

        // swap  
        for (int i = 0; i < n / 2; i++) {
            swap(qubits[i], qubits[n - i - 1]);
        }

        // qubits[0] is LSB, we iterate from 0 up to n-1
        for (int i = 0; i < n; i++) {        
            for (int j = 0; j <= i -1; j++) {
                double angle = -M_PI / (1LL << (i-j)); 
                r1<cudaq::ctrl>(angle, qubits[j], qubits[i]);
            }
            h(qubits[i]);    
        }
        
    }
};



int main(int argc, char* argv[]) {
    if (argc < 3 || argc > 4) {
        std::cerr << "Usage: ./shor.exe N a [t]\n";
        std::cerr << "  N: number to factor\n";
        std::cerr << "  a: base (coprime to N)\n";
        std::cerr << "  t: optional number of phase estimation qubits (default = n)\n";
        return 1;
    }

    const long long N = std::atoll(argv[1]); 
    const long long a = std::atoll(argv[2]);

    if (std::gcd(a, N) != 1) {
        throw std::invalid_argument(
            "Please choose an 'a' coprime to " + std::to_string(N)
        );
    }

    int n = static_cast<int>(std::floor(std::log2(N))) + 1;

    // Default t = n
    int t = n;
    if (argc == 4) {
        t = std::atoi(argv[3]);
        if (t < n) {
            std::cerr << "Warning: t < n is not recommended.\n";
        }
        if (t <= 0) {
            std::cerr << "Error: t must be positive\n";
            return 1;
        }
        std::cout << "Using provided t = " << t << "\n";
    } else {
        std::cout << "Using default t = " << t << "\n";
    }

    std::cout << "Total number of qubits = " << t+2*n+2 << std::endl; 

    // Pre-compute the phase angles for phi_add(N)
    // N_angles is a vector of size n+1
    std::vector<double> N_angles = compute_phi_add_angles(N,n);

    // Pre-compute the input constants for all the controlled-Ua 
    std::vector<long long> Ua_inputs(t);   
    long long current = a % N; // a^(2^0) mod N = a mod N
    Ua_inputs[0] = current;

    for (int i = 1; i < t; ++i) {
        current = (current * current) % N;  
        Ua_inputs[i] = current; // a^(2^i) mod N
    }

    // Pre-compute all the phi_add(a) phase angles
    std::vector<double> Ua_angles;
    std::vector<double> Ua_inv_angles;
    std::vector<int> angle_starts(t);  // starting indices for different control powers
    int offset = 0; 
    // Loop over all control powers in QPE : 2^0 - 2^(t-1)
    for (int k = 0; k < t; ++k) {
        angle_starts[k] = offset;

        long long base = Ua_inputs[k];

        // Each controlled-Ua gate contains a CMULT. Each CMULT has n sets of phase angles (2^j*base % N where j = 0 ~ n-1)
        for (int j = 0; j < n; ++j) {
            long long A = (base * (1LL << j)) % N;
            std::vector<double> A_angles = compute_phi_add_angles(A, n);
            Ua_angles.insert(Ua_angles.end(), A_angles.begin(), A_angles.end());
        }

        long long base_inv = mod_inverse(base, N);

        // Phase angles for inverse
        for (int j = 0; j < n; ++j) {
            long long A = (base_inv * (1LL << j)) % N;
            std::vector<double> A_angles = compute_phi_add_angles(A, n);
            Ua_inv_angles.insert(Ua_inv_angles.end(), A_angles.begin(), A_angles.end());
        }

        offset += n * (n + 1);  // size per power: n sets * (n+1) angles each
    }

    // Sample Shor's circuit
    int shots_count = 10000;
    auto start = std::chrono::high_resolution_clock::now();
    auto result = cudaq::sample(shots_count, Shor, N, a, t, n, Ua_angles, Ua_inv_angles, N_angles, angle_starts);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "\nMeasurement outcomes: " << std::endl;
    result.dump();

    auto duration = std::chrono::duration<double>(end - start).count();
    std::cout << "Simulation runtime: " << duration << "s\n";

    // Post-processing
    long long threshold = static_cast<long long>(shots_count / (1LL << t)); // discard low-count outcomes
    post_process_shor(result, N, a, t, threshold);

    return 0;
}

// Retruns binary expansion of the given integer
std::vector<int> num2bin(long long value, int n_bits){
    std::vector<int> bits;
    for (int i = n_bits - 1; i >= 0; i--) {
        bits.push_back((value >> i) & 1);  // MSB on the left (first)
    }
    
    return bits;
}

// Computes phase angles for the corresponding Draper QFT adder gate
std::vector<double> compute_phi_add_angles(long long a, int n){
    std::vector<int> a_bits = num2bin(a,n+1);
    std::vector<double> a_angles;

    for (int i=0; i < n+1; i++){ // loop from LSB
        double angle = 0;
        std::span<int> slice = std::span(a_bits).subspan(i);
        for (int j=0; j < n+1-i; j++){
            if (slice[j] == 1){
                angle += 2.0L * M_PI/(1LL << (j+1));
           }
        }
        a_angles.push_back(angle);
    }
    return a_angles;
}

// Extended Euclidean Algorithm
// Returns {gcd, x, y} such that a*x + b*y = gcd(a,b)
std::pair<long long, long long> extended_gcd(long long a, long long b){
    if (b == 0) {
        return {1, 0};  // gcd = a, x = 1, y = 0
    }
    auto [x1, y1] = extended_gcd(b, a % b);
    long long x = y1;
    long long y = x1 - (a / b) * y1;
    return {x, y};
}

// Compute modular inverse of a mod N
// Returns a_inv such that (a * a_inv) % N == 1
long long mod_inverse(long long a, long long N){
    auto [x, y] = extended_gcd(a, N);
    long long gcd = a * x + N * y;  // should be gcd

    if (gcd != 1) {
        throw std::runtime_error("Modular inverse does not exist (a and N are not coprime)");
    }

    // Make result positive and in range [0, N-1]
    long long inv = (x % N + N) % N;
    return inv;
}


void __qpu__ phi_add(cudaq::qview<> qubits, const std::vector<double>& angles, int start){

    for (int i=0; i< qubits.size(); i++){
        if (angles[start + i] != 0) {
                r1(angles[start+i], qubits[i]);
            }
    }
}
void __qpu__ phi_add(cudaq::qview<> qubits, cudaq::qubit& ctrl, const std::vector<double>& angles, int start){

    for (int i=0; i< qubits.size(); i++){
        if (angles[start + i] != 0) {
                r1<cudaq::ctrl>(angles[start+i], ctrl , qubits[i]);
            }
    }
}

void __qpu__ phi_sub(cudaq::qview<> qubits, const std::vector<double>& angles, int start){
    
    for (int i = 0; i < qubits.size(); i++) {
        if (angles[start + i] != 0) {
            // subtraction is just rotating by -angle
            r1(-angles[start+i], qubits[i]);
        }
    }
}
void __qpu__ phi_sub(cudaq::qview<> qubits, cudaq::qubit& ctrl, const std::vector<double>& angles, int start){
    
    for (int i = 0; i < qubits.size(); i++) {
        if (angles[start + i] != 0) {
            // subtraction is just rotating by -angle
            r1<cudaq::ctrl>(-angles[start+i], ctrl, qubits[i]);
        }
    }
}

void __qpu__ CMULT(
                    cudaq::qview<> xreg,
                    cudaq::qview<> qubits,
                    cudaq::qubit& ancilla,
                    const std::vector<double>& a_angles,
                    const std::vector<double>& N_angles,
                    int angle_start
                ){

    int msb_idx = qubits.size()-1;  // cudaq cant process qubits[-1] with -1 as index
    int n = xreg.size();
    
    qft{}(qubits);

    for (int i=0; i < n; i++){
        // angle_start:  global index of the block (t blocks)
        // i * (n+1):  local index within the block  (i=0 ~ n-1) 
        int local_start = i * (n + 1) + angle_start;  // offset into block
        
        // Compute a+b mod N                        
        phi_add(qubits, xreg[i], a_angles, local_start);
        phi_sub(qubits, N_angles, 0);
        iqft{}(qubits);
        cx(qubits[msb_idx],ancilla);  // check if MSB = 1 (meaning a+b < N)
        qft{}(qubits);
        phi_add(qubits, ancilla, N_angles, 0);
        
        // Uncompute ancilla
        phi_sub(qubits, xreg[i], a_angles, local_start);
        iqft{}(qubits);
        // Check if MSB =0 (meaning (a+b) (mod N) -a > 0)
        x(qubits[msb_idx]);
        cx(qubits[msb_idx], ancilla);
        x(qubits[msb_idx]);
        qft{}(qubits);
        phi_add(qubits, xreg[i], a_angles, local_start);
    }

    iqft{}(qubits); 
}

void __qpu__ iCMULT(
                    cudaq::qview<> xreg,
                    cudaq::qview<> qubits,
                    cudaq::qubit& ancilla,
                    const std::vector<double>& a_angles,
                    const std::vector<double>& N_angles,
                    int angle_start
                ){

    int msb_idx = qubits.size()-1;
    int n = xreg.size();
    
    qft{}(qubits); 
    
    for (int i=n-1; i >= 0; i--){
        int local_start = i * (n + 1) + angle_start;

        // Uncompute ancilla
        phi_sub(qubits, xreg[i], a_angles, local_start);
        iqft{}(qubits);
        x(qubits[msb_idx]);
        cx(qubits[msb_idx], ancilla);
        x(qubits[msb_idx]);
        qft{}(qubits);
        phi_add(qubits, xreg[i], a_angles, local_start);

        // Compute a+b mod N   
        phi_sub(qubits, ancilla, N_angles, 0);
        iqft{}(qubits);
        cx(qubits[msb_idx],ancilla);
        qft{}(qubits);
        phi_add(qubits, N_angles, 0);
        phi_sub(qubits, xreg[i], a_angles, local_start);
    }

    iqft{}(qubits);
}

void __qpu__ Ua(
                cudaq::qview<> xreg,
                cudaq::qview<> qubits,
                cudaq::qubit& ancilla,
                const std::vector<double>& a_angles,
                const std::vector<double>& a_inv_angles,
                const std::vector<double>& N_angles,
                int angle_start
            ){
    
    CMULT(xreg, qubits, ancilla, a_angles, N_angles, angle_start);
    for (int i=0; i < xreg.size(); i++){
        swap(xreg[i],qubits[i]);  // Do not swap the msb of qubits
    }
    iCMULT(xreg, qubits, ancilla, a_inv_angles, N_angles, angle_start);
}

void __qpu__ Shor(
                long long N,
                long long a,
                int t,
                int n,
                const std::vector<double>& Ua_angles,
                const std::vector<double>& Ua_inv_angles,
                const std::vector<double>& N_angles,
                const std::vector<int>& angle_starts
                ){

    cudaq::qvector phase(t);
    cudaq::qvector xreg(n);
    cudaq::qvector qubits(n+1);
    cudaq::qubit ancilla;

    h(phase);
    x(xreg[0]); // Initial |x> = |1>

    for (int i=0; i < t; i++){
        // control(kernel, control qubits, kernel arguments)
        cudaq::control(Ua,phase[i],xreg,qubits,ancilla,Ua_angles,Ua_inv_angles,N_angles,angle_starts[i]);  
    }

    iqft{}(phase);

    mz(phase);
}


// Continued fractions 
std::set<long long> continued_fractions(double phase, long long Q, long long max_den = 0){
    
    std::set<long long> cands;
    if (max_den == 0) max_den = 2 * Q;  // default to 2*Q
    
    if (phase < 1e-12) return cands;
    
    // Method 1: Best fraction approximation 
    long long best_num = 0, best_den = 1;
    double best_error = std::abs(phase);
    
    for (long long den = 1; den <= max_den; den++) {
        long long num = static_cast<long long>(std::round(phase * den));
        double error = std::abs(phase - static_cast<double>(num) / den);
        if (error < best_error) {
            best_error = error;
            best_num = num;
            best_den = den;
        }
    }
    
    if (best_den > 1 && std::abs(phase - static_cast<double>(best_num) / best_den) < 1.0 / (2.0 * Q)) {
        cands.insert(best_den);
    }
    
    // Method 2: Full CF expansion
    double x = phase;
    long long h = 0, k = 1;  // previous numerator, previous denominator
    long long h1 = 1, k1 = 0; // current numerator, current denominator
    
    const int MAX_ITER = 20;
    for (int i = 0; i < MAX_ITER && x > 1e-12; ++i) {
        long long a = static_cast<long long>(std::floor(x));
        
        // Update numerator and denominator
        long long h2 = a * h1 + h;
        long long k2 = a * k1 + k;
        
        long long p = h2, q = k2;
        
        // Check if this convergent is good
        if (q > 1 && q <= max_den) {
            double approx = static_cast<double>(p) / q;
            if (std::abs(phase - approx) < 1.0 / (2.0 * Q)) {
                cands.insert(q);
            }
        }
        
        h = h1; k = k1;
        h1 = h2; k1 = k2;
        
        x -= a;
        if (x > 1e-12) x = 1.0 / x;
    }
    
    return cands;
}

void post_process_shor(
    const cudaq::sample_result& result,  
    long long N,
    long long a,
    int t,
    int min_shots_threshold
) {
    long long Q = 1LL << t;
    
    // 1. Collect measurement outcomes
    std::map<long long, long long> x_counts;
    long long total_shots = 0;
    
    for (const auto& [bitstring, count] : result) {
        long long x = 0;
        for (size_t i = 0; i < bitstring.size(); ++i) {
            if (bitstring[i] == '1') {
                x |= (1LL << i);  
            }
        }
        x_counts[x] += count;
        total_shots += count;
    }
    
    
    // 2. Collect r candidates weighted by counts  
    std::map<long long, long long> r_candidates;
    
    // Use provided threshold for filtering 
    int processed = 0;
    for (const auto& [x, cnt] : x_counts) {
        if (cnt < min_shots_threshold) continue;
        
        double phase = static_cast<double>(x) / Q;
        
        // Get candidate periods from continued fractions 
        auto denoms = continued_fractions(phase, Q, std::max(Q, N));
        
        for (long long r : denoms) {
            r_candidates[r] += cnt;
        }
        processed++;
    }
    
    if (r_candidates.empty()) {
        std::cout << "\nNo period candidates found.\n";
        return;
    }
    
    // 3. Validate: even r and a^r mod N = 1 
    std::vector<std::pair<long long, long long>> valid_r;
    
    for (const auto& [r, weight] : r_candidates) {
        if (r % 2 != 0) continue;  // Skip odd periods
        
        // Check a^r mod N
        long long pow_ar = 1;
        long long base = a % N;
        long long exp = r;
        while (exp > 0) {
            if (exp & 1) pow_ar = (pow_ar * base) % N;
            base = (base * base) % N;
            exp >>= 1;
        }
        
        if (pow_ar == 1) {
            valid_r.emplace_back(r, weight);
        } 
    }
    
    if (valid_r.empty()) {
        std::cout << "\nNo valid r found. Try larger t or different 'a'.\n";
        return;
    }
    
    // 4. Use smallest valid r 
    std::sort(valid_r.begin(), valid_r.end());
    long long p = valid_r[0].first;
    
    std::cout << "\nSelected period p = " << p << "\n";
    
    // 5. Test all even divisors d|p for factors
    std::set<std::pair<long long, long long>> all_factors;
    
    // Find all even divisors of p
    std::vector<long long> divisors;
    for (long long d = 2; d <= p; d += 2) {
        if (p % d == 0) {
            divisors.push_back(d);
        }
    }
    
    for (long long d : divisors) {
        long long exp = d / 2;
        
        // Compute a^exp mod N
        long long a_pow = 1;
        long long base = a % N;
        long long e = exp;
        while (e > 0) {
            if (e & 1) a_pow = (a_pow * base) % N;
            base = (base * base) % N;
            e >>= 1;
        }
        
        long long f1 = std::gcd(a_pow - 1, N);
        long long f2 = std::gcd(a_pow + 1, N);
        
        if (f1 > 1 && f1 < N) {
            long long other = N / f1;
            all_factors.emplace(std::min(f1, other), std::max(f1, other));
        }
        if (f2 > 1 && f2 < N) {
            long long other = N / f2;
            all_factors.emplace(std::min(f2, other), std::max(f2, other));
        }
    }
    
    // 6. Final results
    std::cout << "\n=== Factorization Result ===\n";
    if (all_factors.empty()) {
        std::cout << "Only trivial factors found. Try different 'a'.\n";
    } else {
        // Pick smallest non-trivial factor pair
        auto [factor1, factor2] = *all_factors.begin();
        std::cout << N << " = " << factor1 << " x " << factor2 << "\n";
        
    }
}
