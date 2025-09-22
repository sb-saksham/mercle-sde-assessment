# Privacy Prototype For Mercle Problem Statement
### System Requirements and Perquisites:
- OpenFHE C++ Library Installed
- Tested On Ubuntu 22 (WSL)
- 12GB Ram
### Steps to Install
1. ` git clone https://github.com/sb-saksham/mercle-sde-assessment.git`
2. `pip install -r requirements.txt`
3. Install `openfhe-python` using `pip install openfhe` or build from [source](https://github.com/openfheorg/openfhe-python?tab=readme-ov-file#building-from-source).
4. Run the file `python main.py`
## Overview of Prototype
### Utils
1. `def  get_sign_poly_coeffs(degree=3, domain=(-2.0, 2.0)):` Used to return coefficients for the sign function using the given degree of polynomial.
	Currently 15 degree polynomial coefficients are used.
2. `def  generate_data(num_vectors, dim):` Generate random database of given size and dimensions.
3. `class  Party:` Used to define objects representing different parties or nodes which will jointly create a single encryption key, preventing secret key to be with a single person/party.
### Privacy Prototype Class
The `class  PrivacyEngine:` with methods to encrypt, decrypt, run computation, and do the threshold check.
##### Methods
- `def  __init__(self, num_parties, ring_dim, mult_depth, scaling_mod_size):`
`self.cc`, `self.joint_public_key`, `self.parties` are the class attributes to be used throughout the execution. The `__init__` method then calls `self._setup_crypto(num_parties, ring_dim, mult_depth, scaling_mod_size)` with the arguments passed to it, which creates the `CryptoContext` with parameters to be used throughout the execution. 

- `def  _setup_crypto(self, num_parties, ring_dim, mult_depth, scaling_mod_size):`
	Set Parameters passed to it to create the `CryptoContext` object, enable features, and then create parties that will be used to create joint keys later.
**In production, the parties will be in different nodes and will give their shards to generate the joint public key.**

- `def  generate_all_eval_keys(self, vector_dim, ring_dim):`
	The function generates the all the evaluation keys required in the whole prototype (Mult, Rotation, etc). Although the joint secret key (generated from all shards is momentarily here) it is not stored beyond the scope of this function.
	**In production environment, this will be the node that does the encryption at the end, receiving all the shards. It will not have the complete key with him only his share which it will provide to generate the joint key and resume the execution.**

- `def  encrypt_data_batched(self, database, query, vector_dim, ring_dim):`
	Returns the database encrypted and batched using the joint public key generated in `generate_all_eval_keys` with the query vector.

- `def  _encrypted_compare(self, ct1, ct2, sign_coeffs):`
	The function homomorphically finds maximum between the two given cipher texts.

- `def  _bootstrap_ciphertext_list(self, ciphertext_list):`
	The function is used to bootstrap ciphertext to refuel its multiplicative depth. After many operations the encrypted data is not operable and runs out of multiplicative depth (argument given while initializing `PrivacyEngine`, to overcome this the ciphertext is re-encrypted to give it a new start with multiplicative depth available to full limit.

- `def  run_computation_batched(self, batched_db_cts, batched_query_cts, vector_dim, ring_dim):`
	This is one of the core function that performs the cosine similarities on pre-encrypted batches and return the max similarity score from the database against the query. It uses the divided batches of db and compute their max scores against query vector, then computes the highest score among all the batches, and finally return the max score.

- `def  check_against_threshold(self, C_max, tau):`
	This function is to be called after running the computation. It returns the encrypted comparison of Tau(the threshold) and max similarity score.

- `def  decrypt_result(self, ciphertext):` This function is used to decrypt the max score to check for absolute error between the encrypted max and actual max.
### `main()`
This function calls all the component and runs the engine sequentially
1. It sets the following parameters:
NUM_VECTORS, DIMENSIONS, TAU, RING_DIM, MULT_DEPTH, SCALING_MOD_SIZE
2. Then it calls the `generate_data(NUM_VECTORS, DIMENSIONS)` to generate the random data.
3. It then Initialize and run `PrivacyEngine(num_parties=3, ring_dim=RING_DIM, mult_depth=MULT_DEPTH, scaling_mod_size=SCALING_MOD_SIZE)` by passing the set parameters, after that call the `engine.generate_all_eval_keys(DIMENSIONS, RING_DIM)` to generate all the evaluation keys. The using these keys the database is encrypted by calling `engine.encrypt_data_batched(db, query, DIMENSIONS, RING_DIM)`. 
4. Then the max similarity is calculated by calling `engine.run_computation_batched(batched_db_cts, batched_query_cts, DIMENSIONS, RING_DIM)` passing the encrypted database.
5. After this for Accuracy Verification, the max similarity is decrypted by calling `engine.decrypt_result(C_max)`, which is then checked against plaintext similarity.
6. At last the threshold check is done by calling `engine.check_against_threshold(C_max, TAU)` and result is displayed.
## OpenFHE Operations Logic
#### Cosine Similarity
This process computes the dot product for unit-normalized vectors, which is equivalent to their cosine similarity. The calculation is batched, meaning multiple dot products are computed in parallel within a single ciphertext. (Function  `run_computation_batched`)
`ct_prod = self.cc.EvalMult(ct_db, ct_query)` 
`rotated = self.cc.EvalRotate(ct_sum, rotation_amount)` 
`ct_sum = self.cc.EvalAdd(ct_sum, rotated)`

For a query `q` and a database vector `v`, both of dimension `d=512`, the cosine similarity is `cos(q, v) = sum(q_i * v_i for i=1 to 512)`.

1.  Element-wise Multiplication: The `EvalMult` operation multiplies the corresponding elements of the batched query and database ciphertexts.
If the plaintext slots for the database are `[v1_1, v1_2, ..., v2_1, v2_2, ...]` and for the query are `[q_1, q_2, ..., q_1, q_2, ...]`, the result is a ciphertext containing the products `[v1_1*q_1, v1_2*q_2, ..., v2_1*q_1, v2_2*q_2, ...]`.
        
2.  Logarithmic Summation (using Rotations): To sum the 512 product terms for each vector, the code uses rotations. A rotation is a cyclic shift of the values in the ciphertext's slots.
Let the ciphertext `C` contain the products `[p_1, p_2, ..., p_512]`.
	- a. Rotate `C` by 1 slot to get `C_rot1 = [p_2, ..., p_512, p_1]`. Add them: `C_sum1 = C + C_rot1`. The first slot now contains `p_1 + p_2`.
        
	- b. Rotate `C_sum1` by 2 slots and add. The first slot now contains `(p_1 + p_2) + (p_3 + p_4)`.

	This process is repeated `log2(512) = 9` times with rotation amounts of `1, 2, 4, 8, ... 256`. At the end, the first slot for each vector's block contains the full sum of its 512 products, which is the final dot product score.

#### Global Maximum
This is a two-part process: first, defining a way to compare two encrypted numbers, and second, applying that comparison in a tournament-style reduction to find the maximum among many scores.
##### 1. Comparing Two Encrypted Numbers (`max(a, b)`)

Function name `_encrypted_compare`
`diff = self.cc.EvalSub(ct1, ct2)`
`sign_of_diff = self.cc.EvalPoly(diff, sign_coeffs)`
`result = self.cc.EvalMult(sum_terms, 0.5)`
The function homomorphically computes `max(a, b)` using the mathematical identity: `max(a, b) = 0.5 * (a + b + sign(a - b) * (a - b))`

1.  Compute the difference: `EvalSub` calculates the encrypted difference `d = a - b`.
    
2.  Approximate sign: `EvalPoly` evaluates a high-degree polynomial `P(x)` that approximates the `sign(x)` function. The `sign(x)` function (which is +1 for positive numbers, -1 for negative) is not a simple arithmetic operation, so it must be approximated.
    
    -  `sign_d ≈ P(d) = c15*d^15 + c14*d^14 + ... + c0`. This is the most computationally intensive step.
        
3.  Combine terms: The final result is assembled according to the formula using `EvalAdd` and `EvalMult`. The logic holds: if `a > b`, then `sign(a-b)` is `+1`, and the formula simplifies to `0.5 * (2*a) = a`. If `b > a`, `sign(a-b)` is `-1`, and it simplifies to `b`.

##### 2. Finding the Global Max from Many Numbers
Function name `run_computation_batched`
`while len(scores) > 1: ... max_val = self._encrypted_compare(...)`
 `for i in range(int(math.log2(batch_size))): ... temp_max = self._encrypted_compare(...)`

1.  Inter-Batch Reduction: The `while` loop compares encrypted scores pairwise. If you have 4 ciphertexts `[C1, C2, C3, C4]`, the first round computes `max(C1, C2)` and `max(C3, C4)`, leaving 2 ciphertexts. The next round computes the max of those two, leaving the final winner.
This reduces `N` candidates to `1` in `log2(N)` rounds of comparison.
        
2.  Intra-Batch Reduction (using Rotations): The final winning ciphertext still contains multiple candidate scores packed inside it (e.g., `[max_v1, max_v2, ..., max_v256]`). The `for` loop finds the single max within this ciphertext.
Similar to the summation, it rotates the ciphertext by `vector_dim * 2^i` slots and compares.
        
    1. Rotate `C` by 512. Now `slot_0` is compared with `slot_512`, `slot_1` with `slot_513`, etc., in parallel. The ciphertext is updated with the pairwise maximums.
        
    2. Rotate the result by `512 * 2 = 1024` and compare again.
        
	After `log2(batch_size)` steps, the score in the very first slot (`slot_0`) is the true global maximum.
#### Threshold Calculation
This determines if the maximum similarity score is greater than a threshold `tau` without decrypting the score.
Function name `check_against_threshold`
 `C_diff = self.cc.EvalSub(C_max, tau)` 
 `C_sign = self.cc.EvalPoly(C_diff, sign_coeffs)`

The goal is to compute `sign(C_max - tau)`.

1. Compute the difference: `EvalSub` subtracts the public threshold value `tau` from the encrypted maximum score `C_max`.
 `diff = C_max - tau`.
        
2.  Evaluate sign polynomial: `EvalPoly` is used again to compute the sign of this difference.
    `result = sign(diff)`.
        
    -   If `C_max > tau`, `diff` is positive, and the decrypted result will be `+1`.
        
    -   If `C_max <= tau`, `diff` is negative or zero, and the decrypted result will be `-1` or `0`.
    
#### Bootstrapping of Ciphertext

Bootstrapping is a process to "refresh" a ciphertext. It reduces the computational noise that accumulates after many operations (especially multiplications), effectively resetting the multiplicative depth and allowing for deeper computations.
Function Name `_bootstrap_ciphertext_list`
`ct_adj = self.cc.IntMPBootAdjustScale(ct)` 
`shares = self.cc.IntMPBootDecrypt(party.kpShard.secretKey, c1, a)` `aggregated_shares = self.cc.IntMPBootAdd(share_pair_vec)` 
`refreshed_ct = self.cc.IntMPBootEncrypt(...)`

It performs a homomorphic decryption and re-encryption in an interactive protocol without ever revealing the plaintext.
    -   A noisy ciphertext `Enc(m)` is prepared (`IntMPBootAdjustScale`).
    -   The parties collaboratively generate a set of decryption shares for this ciphertext using their secret key shards (`IntMPBootDecrypt`). These shares are themselves encrypted or protected.
    -   The shares are aggregated (`IntMPBootAdd`). The aggregated share is not enough to reveal the message `m` but contains the necessary information to reconstruct it.
    -   Finally, this aggregated information is used to generate a new, fresh ciphertext of the same message `m` (`IntMPBootEncrypt`). This new ciphertext has low noise and a full multiplicative budget.
 
In short, it's a secure, multiparty procedure to take a "noisy" `Enc(m)` and produce a "clean" `Enc(m)`.

#### Factors
- Ring Dimension `RING_DIM`: 
The number of plaintext values you can pack into a single ciphertext (`slots = RING_DIM / 2`). Your code uses a large value of `1 << 18` (262,144), which provides 131,072 slots. This is crucial for **performance**, as it allows you to batch 256 vectors (131,072 / 512) at once.
Higher ring dimension means higher cryptographic security level.

- Scaling Factor (`SCALING_MOD_SIZE`):
`SCALING_MOD_SIZE` sets the size of this factor. A larger value, like the `58` bits in your code, means a larger `Δ`, which allows you to encode numbers with higher precision. This increases  
- Multiplicative Depth (`MULT_DEPTH`)
It defines the computational budget for your encrypted operations, specifically for multiplications.
Each multiplication increases the noise in a ciphertext. `MULT_DEPTH` determines the maximum number of sequential multiplications you can perform.
#### How to Correct Error
-   Scaling Modulus Size (`SCALING_MOD_SIZE`):
	Increasing this value makes the initial scaling factor larger. This means your floating-point numbers are encoded with more bits of precision. Since each multiplication consumes some of this precision, starting with a larger budget ensures that even after many operations, the final decrypted value is much closer to the true plaintext result. 
- Multiplicative Depth (`MULT_DEPTH`) 
	The multiplicative depth is your computational budget. An insufficient depth is a primary cause of incorrect results and large errors. For high number of computations while keeping noise low and accuracy high, high multiplicative depth
- Polynomial Degree
The `sign` function approximation could be reduced by increasing the polynomial degree.
A higher-degree polynomial (e.g., degree 15, 31, or even 63) provides a much closer mathematical fit to the true `sign(x)` function. This directly reduces the _approximation error_, which might be a major source of inaccuracy in the homomorphic `max` calculation. But a higher degree requires a larger multiplicative depth, making the computation more resource-intensive.

#### Scaling Plan 1M
- ##### Sharding:
	- Split the 1 million encrypted vectors into smaller, manageable chunks. For example, you could create 100 shards, each containing 10,000 encrypted vectors.
    
	- Storage: These encrypted shards would be stored in a distributed, high-throughput storage system like a key-value store (e.g., Redis) or a distributed filesystem, not on a single local disk. This allows for fast, parallel access.
- ##### Parallel Computation
	With the data sharded, you can now process it in parallel across multiple compute nodes (workers).

	- A fleet of worker nodes is spun up. Each worker is assigned one or more shards of the database. The client's single encrypted query vector is broadcast to all workers.

	- Each worker independently executes the exact same logic as the prototype on its local data shard. It computes the batched dot products and performs the tournament-style reduction to find the local maximum similarity score for its specific shard. Because the workers don't need to communicate with each other, this step scales horizontally (100 workers for 100 shards).
- ##### Hierarchical Reduction
	After the parallel phase, one encrypted local maximum is left from each of the 100 workers. The final step is to find the single global maximum among them.

	- The 100 encrypted local maximum ciphertexts are sent from the workers to a central aggregator node.
    
	- This aggregator node performs a final, small-scale homomorphic tournament on the 100 incoming ciphertexts. It uses the same `_encrypted_compare` function to find the one true global maximum value. This final computation is extremely fast compared to processing the entire dataset.

#### Notes
- Key Management: The multiparty key management would ideally work as if there are various nodes being the secure key-holding parties that would only interact with the final aggregator node to perform the multiparty decryption of the single, final global maximum ciphertext.

