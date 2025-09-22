import time
import numpy as np
from openfhe import *
import math

# Helper Functions

# Polynomial Approximation for the Sign Function
def get_sign_poly_coeffs(degree=3, domain=(-2.0, 2.0)):
    """
    Generates coefficients for a Chebyshev polynomial approximation of the
    sign function over a specified domain.
    """
    # Using pre-computed coefficients for stability and speed.
    # These were generated using the commented-out numpy code.
    # sign_func = np.sign
    # chebyshev_poly = np.polynomial.chebyshev.Chebyshev.fit(
    #     np.linspace(domain[0], domain[1], 200),
    #     sign_func(np.linspace(domain[0], domain[1], 200)),
    #     deg=15 # A higher degree was used for better accuracy
    # )
    # power_series_poly = chebyshev_poly.convert(kind=np.polynomial.polynomial.Polynomial)
    # return power_series_poly.coef.tolist()
    return [6.053, 5.237, -3.664, -19.165, 8.489, 35.944, -1.074, -34.877, 7.563, 18.690, -2.850, -5.586, 5.327, 0.872, -3.867, -0.055]


# Data Generation 
def generate_data(num_vectors, dim):
    """Generates a random database and query, both unit-normalized."""
    # Generate random data from a standard normal distribution
    db = np.random.randn(num_vectors, dim).astype(np.float64)
    query = np.random.randn(1, dim).astype(np.float64)
    
    # Normalize each vector to have a unit L2 norm
    db /= np.linalg.norm(db, axis=1)[:, np.newaxis]
    query /= np.linalg.norm(query, axis=1)[:, np.newaxis]
    
    return db.tolist(), query[0].tolist()

class Party:
    def __init__(self, id=None, sharesPair=None, kpShard=None):
        self.id = id
        self.sharesPair = sharesPair
        self.kpShard = kpShard

    def __str__(self):
        return f"Party {self.id}"

class PrivacyEngine:
    def __init__(self, num_parties, ring_dim, mult_depth, scaling_mod_size):
        self.cc = None
        self.joint_public_key = None
        self.parties = None
        self._setup_crypto(num_parties, ring_dim, mult_depth, scaling_mod_size)

    def _setup_crypto(self, num_parties, ring_dim, mult_depth, scaling_mod_size):
        print("Setting up CryptoContext")

        # Create CryptoContext
        parameters = CCParamsCKKSRNS()
        parameters.SetRingDim(ring_dim)
        parameters.SetScalingModSize(scaling_mod_size)
        parameters.SetMultiplicativeDepth(mult_depth)
        parameters.SetInteractiveBootCompressionLevel(COMPRESSION_LEVEL.SLACK)

        self.cc = GenCryptoContext(parameters)

        # Enable features individually
        self.cc.Enable(PKESchemeFeature.PKE)
        self.cc.Enable(PKESchemeFeature.KEYSWITCH)
        self.cc.Enable(PKESchemeFeature.LEVELEDSHE)
        self.cc.Enable(PKESchemeFeature.ADVANCEDSHE)
        self.cc.Enable(PKESchemeFeature.MULTIPARTY)
        print("CryptoContext setup complete.\n")
        self.parties = [Party() for _ in range(num_parties)]
        for i in range(num_parties):
            self.parties[i].id = i
            print(f"Party {self.parties[i].id} started.")
            if i == 0:
                self.parties[i].kpShard = self.cc.KeyGen()
            else:
                self.parties[i].kpShard = self.cc.MultipartyKeyGen(self.parties[0].kpShard.publicKey)
            print(f"Party {i} key generation completed.\n")
        
        # Assert everything is good
        for i in range(num_parties):
            if not self.parties[i].kpShard.good():
                print(f"Key generation failed for party {i}!\n")
                return 1
        print("Created Joint Public Key with different shares/parties")

    def generate_all_eval_keys(self, vector_dim, ring_dim):
        """
        Performs a secure ceremony to generate the joint public key and all
        necessary evaluation keys (Mult, Sum, Rotate). The joint secret key
        is ephemeral and exists only within this function's scope.
        """
        print("Starting Key Generation")
        
        # Gather all key shares/shards from the parties
        secret_key_shares = [p.kpShard.secretKey for p in self.parties]

        # Create the joint key pair.
        kpMultiparty = self.cc.MultipartyKeyGen(secret_key_shares)
        
        # Store the public key. This is safe to keep.
        self.joint_public_key = kpMultiparty.publicKey
        
        # Use the ephemeral joint secret key to generate ALL evaluation keys.
        print("Generating multiplication keys...")
        self.cc.EvalMultKeyGen(kpMultiparty.secretKey)
        
        print("Generating rotation keys...")
        slots = ring_dim // 2
        batch_size = slots // vector_dim
        summation_indices = [1 << i for i in range(int(math.log2(vector_dim)))]
        reduction_indices = [vector_dim * (1 << i) for i in range(int(math.log2(batch_size)))]
        all_indices = list(set(summation_indices + reduction_indices))
        self.cc.EvalRotateKeyGen(kpMultiparty.secretKey, all_indices)
        
        print(f"Generated {len(all_indices)} unique rotation keys.")
        print("Key Generation Complete\n")

    def encrypt_data_batched(self, database, query, vector_dim, ring_dim):
        """
        Packs plaintext vectors into batches and encrypts each batch.
        """
        print("Encrypting database and query in batches...")
        slots = ring_dim // 2
        batch_size = slots // vector_dim
        num_vectors = len(database)

        batched_db_cts = []
        batched_query_cts = []

        for i in range(0, num_vectors, batch_size):
            db_batch_list = database[i : i + batch_size]
            current_batch_size = len(db_batch_list)

            # Pad the plaintext vectors for encryption
            db_plaintext_batch = [val for vec in db_batch_list for val in vec]
            query_plaintext_batch = query * current_batch_size

            # Encrypt the prepared batches
            ct_db = self.cc.Encrypt(self.joint_public_key, self.cc.MakeCKKSPackedPlaintext(db_plaintext_batch))
            ct_query = self.cc.Encrypt(self.joint_public_key, self.cc.MakeCKKSPackedPlaintext(query_plaintext_batch))

            batched_db_cts.append(ct_db)
            batched_query_cts.append(ct_query)

        print(f"Encrypted {num_vectors} vectors into {len(batched_db_cts)} database ciphertexts.\n")
        return batched_db_cts, batched_query_cts

    def _encrypted_compare(self, ct1, ct2, sign_coeffs):
        """
        Homomorphically computes max(ct1, ct2).
        """
        diff = self.cc.EvalSub(ct1, ct2)
        sign_of_diff = self.cc.EvalPoly(diff, sign_coeffs)
        
        term1 = self.cc.EvalAdd(ct1, ct2)
        term2 = self.cc.EvalMult(sign_of_diff, diff)
        
        sum_terms = self.cc.EvalAdd(term1, term2)
        result = self.cc.EvalMult(sum_terms, 0.5)
        
        return result
    def _bootstrap_ciphertext_list(self, ciphertext_list):
        """
        Refreshes a list of ciphertexts using the interactive multiparty bootstrapping protocol.
        """
        refreshed_list = []
        num_to_bootstrap = len(ciphertext_list)
        
        for i, ct in enumerate(ciphertext_list):
            print(f"     - Bootstrapping ciphertext {i+1}/{num_to_bootstrap}...")
            
            ct_adj = self.cc.IntMPBootAdjustScale(ct)
            a = self.cc.IntMPBootRandomElementGen(self.parties[0].kpShard.publicKey)
            share_pair_vec = []
            c1 = ct_adj.Clone()
            c1.RemoveElement(0) 

            for party in self.parties:
                shares = self.cc.IntMPBootDecrypt(party.kpShard.secretKey, c1, a)
                share_pair_vec.append(shares)

            aggregated_shares = self.cc.IntMPBootAdd(share_pair_vec)

            refreshed_ct = self.cc.IntMPBootEncrypt(
                self.joint_public_key, aggregated_shares, a, ct_adj
            )
            refreshed_list.append(refreshed_ct)

        return refreshed_list
        
    def run_computation_batched(self, batched_db_cts, batched_query_cts, vector_dim, ring_dim):
        """
        Computes cosine similarities on pre-encrypted batches and finds the max score.
        """
        print("Running Batched Dot Product Computation...")

        slots = ring_dim // 2
        batch_size = slots // vector_dim
        
        print(f"   - Ciphertext slots: {slots}")
        print(f"   - Vector dimension: {vector_dim}")
        print(f"   - Batch size (vectors per ciphertext): {batch_size}")

        sign_coeffs = get_sign_poly_coeffs()
        batched_similarities = []

        print("\nComputing encrypted dot products from given batches...")
        for i, (ct_db, ct_query) in enumerate(zip(batched_db_cts, batched_query_cts)):
            ct_prod = self.cc.EvalMult(ct_db, ct_query)
            
            # Sum products using MEMORY-EFFICIENT logarithmic summation
            ct_sum = ct_prod
            for j in range(int(math.log2(vector_dim))):
                rotation_amount = 1 << j
                rotated = self.cc.EvalRotate(ct_sum, rotation_amount)
                ct_sum = self.cc.EvalAdd(ct_sum, rotated)
            
            batched_similarities.append(ct_sum)
            print(f"   - Processed batch {i+1}/{len(batched_db_cts)}")

        print(f"Computed {len(batched_similarities)} batched similarity ciphertexts.")
        
        print("\nRefreshing ciphertexts via bootstrapping to restore depth...")
        batched_similarities = self._bootstrap_ciphertext_list(batched_similarities)
        print("Bootstrapping complete.\n")

        print("\nFinding maximum score (inter-batch reduction)...")
        scores = batched_similarities
        while len(scores) > 1:
            next_round_scores = []
            # Refresh before each round of comparisons to maintain precision
            scores = self._bootstrap_ciphertext_list(scores) 
            for i in range(0, len(scores), 2):
                if i + 1 < len(scores):
                    max_val = self._encrypted_compare(scores[i], scores[i+1], sign_coeffs)
                    next_round_scores.append(max_val)
                else:
                    next_round_scores.append(scores[i])
            scores = next_round_scores
            print(f"   - Reduction round complete. {len(scores)} ciphertexts remaining.")

        final_batch_max = scores[0]

        # Finding single max from final batch using LOGARITHMIC REDUCTION 
        print("\nFinding single max from final batch using logarithmic reduction...")
        temp_max = final_batch_max
        for i in range(int(math.log2(batch_size))):
            rotation_amount = vector_dim * (1 << i)
            rotated = self.cc.EvalRotate(temp_max, rotation_amount)
            # Compare and update the max in parallel across the ciphertext
            temp_max = self._encrypted_compare(temp_max, rotated, sign_coeffs)

        C_max = temp_max # The global max is now in the first slot
        print("Encrypted maximum value computed.\n")
        return C_max

    def check_against_threshold(self, C_max, tau):
        print(f"Performing threshold check against tau = {tau}...")
        C_diff = self.cc.EvalSub(C_max, tau)
        
        sign_coeffs = get_sign_poly_coeffs()
        C_sign = self.cc.EvalPoly(C_diff, sign_coeffs)
        print("Encrypted threshold check complete.\n")
        return C_sign

    def decrypt_result(self, ciphertext):
        """
        Performs a multiparty decryption of a ciphertext.
        """
        print("Performing multiparty decryption...")
        
        partial_decryptions = []

        lead_secret_key = self.parties[0].kpShard.secretKey
        decryption_lead = self.cc.MultipartyDecryptLead([ciphertext], lead_secret_key)
        partial_decryptions.append(decryption_lead[0])
        print("   - Lead party (Party 0) provided partial decryption.")

        for i in range(1, len(self.parties)):
            main_secret_key = self.parties[i].kpShard.secretKey
            decryption_main = self.cc.MultipartyDecryptMain([ciphertext], main_secret_key)
            partial_decryptions.append(decryption_main[0])
            print(f"   - Main party (Party {i}) provided partial decryption.")

        plaintext = self.cc.MultipartyDecryptFusion(partial_decryptions)

        print("Decryption complete.")
        plaintext.SetLength(8)
        return plaintext.GetCKKSPackedValue()


def main():
    start_time = time.time()
    
    # System Configuration
    NUM_VECTORS = 4
    DIMENSIONS = 512
    TAU = 0.85
    RING_DIM = 1 << 18
    MULT_DEPTH = 60
    SCALING_MOD_SIZE = 58

    # Generate data
    db, query = generate_data(NUM_VECTORS, DIMENSIONS)
    
    # Initialize and run
    engine = PrivacyEngine(num_parties=3, ring_dim=RING_DIM, mult_depth=MULT_DEPTH, scaling_mod_size=SCALING_MOD_SIZE)
    
    engine.generate_all_eval_keys(DIMENSIONS, RING_DIM)

    batched_db_cts, batched_query_cts = engine.encrypt_data_batched(db, query, DIMENSIONS, RING_DIM)
    
    C_max = engine.run_computation_batched(batched_db_cts, batched_query_cts, DIMENSIONS, RING_DIM)
    
    # Accuracy Verification
    print("Accuracy Verification")
    decrypted_max_result = engine.decrypt_result(C_max)
    he_max_similarity = decrypted_max_result[0].real
    
    plaintext_similarities = [np.dot(db_vec, query) for db_vec in db]
    plaintext_max_similarity = max(plaintext_similarities)
    
    error = abs(he_max_similarity - plaintext_max_similarity)
    
    print(f"Plaintext Max Similarity:  {plaintext_max_similarity:.8f}")
    print(f"Decrypted HE Max Similarity: {he_max_similarity:.8f}")
    print(f"Absolute Error:          {error:.8f}")
    
    if error < 1e-4:
        print("Accuracy target (< 10^-4) met!\n")
    else:
        print("Accuracy target not met.\n")
    
    # THreshold Check
    C_sign = engine.check_against_threshold(C_max, TAU)
    
    decrypted_sign_result = engine.decrypt_result(C_sign)
    final_sign = decrypted_sign_result[0].real
    
    print("Final Decision")
    if final_sign > 0:
        print(f"Decision: NOT-UNIQUE (Max similarity > {TAU})\n")
    else:
        print(f"Decision: UNIQUE (Max similarity <= {TAU})\n")

    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()