import time
import numpy as np
from openfhe import *
import math

# ==============================================================================
# 0. Helper Functions and Configuration
# ==============================================================================

# --- Polynomial Approximation for the Sign Function ---
def get_sign_poly_coeffs(degree=2, domain=(-2.0, 2.0)):
    # """
    # Generates coefficients for a Chebyshev polynomial approximation of the
    # sign function over a specified domain.
    # """
    # # The function to approximate is sign(x)
    # sign_func = np.sign

    # # Generate the Chebyshev approximation
    # chebyshev_poly = np.polynomial.chebyshev.Chebyshev.fit(
    #     np.linspace(domain[0], domain[1], 200), # Sample points in domain
    #     sign_func(np.linspace(domain[0], domain[1], 200)), # sign(x) at points
    #     deg=degree
    # )
    # # Convert Chebyshev coefficients to standard power series coefficients
    # power_series_poly = chebyshev_poly.convert(kind=np.polynomial.polynomial.Polynomial)
    # return power_series_poly.coef.tolist()
    # For testing
    return [2.93, -5.9, 3.8]



# --- Data Generation ---

def generate_data(num_vectors, dim):
    """Generates a random database and query, both unit-normalized."""
    
    # Generate random data from a standard normal distribution
    db = np.random.randn(num_vectors, dim).astype(np.float64)
    query = np.random.randn(1, dim).astype(np.float64)

    # Normalize each vector to have a unit L2 norm
    db /= np.linalg.norm(db, axis=1)[:, np.newaxis]
    query /= np.linalg.norm(query, axis=1)[:, np.newaxis]

    return db.tolist(), query[0].tolist()


def _encrypted_compare(cc, ct1, ct2, sign_coeffs):
    """(Unchanged) Homomorphically computes max(ct1, ct2)."""

    diff = cc.EvalSub(ct1, ct2)
    sign_of_diff = cc.EvalPoly(diff, sign_coeffs)
    term1 = cc.EvalAdd(ct1, ct2)
    term2 = cc.EvalMult(sign_of_diff, diff)
    cc.ModReduceInPlace(term2)
    sum_terms = cc.EvalAdd(term1, term2)
    result = cc.EvalMult(sum_terms, 0.5)
    cc.ModReduceInPlace(result)
    return result



class Party:

    def __init__(self, id, sharesPair, kpShard):
        self.id = id
        self.sharesPair = sharesPair
        self.kpShard = kpShard
    
    def __init__(self):
        self.id = None
        self.sharesPair = None
        self.kpShard = None
    
    def __str__(self):
        return f"Party {self.id}"

# Main Engine

class PrivacyEngine:
    def __init__(self, num_parties, ring_dim, mult_depth, scaling_mod_size):
        self.cc = None
        self.joint_public_key = None
        self.parties = None
        self._setup_crypto(num_parties, ring_dim, mult_depth, scaling_mod_size)

    def _setup_crypto(self, num_parties, ring_dim, mult_depth, scaling_mod_size):
        print("Step 1: Setting up CryptoContext and generating multiparty keys")
        # Create CryptoContext 
        parameters = CCParamsCKKSRNS()
        parameters.SetRingDim(ring_dim)
        parameters.SetScalingModSize(scaling_mod_size)
        parameters.SetMultiplicativeDepth(mult_depth)
        self.cc = GenCryptoContext(parameters)

        # Enable features individually
        self.cc.Enable(PKESchemeFeature.PKE)
        self.cc.Enable(PKESchemeFeature.KEYSWITCH)
        self.cc.Enable(PKESchemeFeature.LEVELEDSHE)
        self.cc.Enable(PKESchemeFeature.ADVANCEDSHE)
        self.cc.Enable(PKESchemeFeature.MULTIPARTY)
        print("CryptoContext setup complete.\n")
        self.parties = [Party()] * num_parties
        for i in range(num_parties):

            #define id of parties[i] as i
            self.parties[i].id = i
            print(f"Party {self.parties[i].id} started.")
            if i == 0:
                self.parties[i].kpShard = self.cc.KeyGen()
            else:
                self.parties[i].kpShard = self.cc.MultipartyKeyGen(self.parties[0].kpShard.publicKey)

            print(f"Party {i} key generation completed.\n")
        print("Joint public key for (s_0 + s_1 + ... + s_n) is generated...")
        
        # Assert everything is good
        for i in range(num_parties):
            if not self.parties[i].kpShard.good():
                print(f"Key generation failed for party {i}!\n")
                return 1

        # Generate collective public key (This will happen in final server with the collective secret share)

        secretKeys = []
        for i in range(num_parties):
            secretKeys.append(self.parties[i].kpShard.secretKey)

        kpMultiparty = self.cc.MultipartyKeyGen(secretKeys)
        self.joint_public_key = kpMultiparty.publicKey
        self.cc.EvalMultKeyGen(kpMultiparty.secretKey)
        self.cc.EvalSumKeyGen(kpMultiparty.secretKey)
        print("Created Joint Public Key with different shares/parties")

    def encrypt_data(self, database, query):
        print("Step 2: Encrypting database and query...")
        encrypted_db = [self.cc.Encrypt(self.joint_public_key, self.cc.MakeCKKSPackedPlaintext(vec)) for vec in database]
        encrypted_query = self.cc.Encrypt(self.joint_public_key, self.cc.MakeCKKSPackedPlaintext(query))
        print(f"✅ Encrypted {len(encrypted_db)} database vectors and 1 query vector.\n")
        return encrypted_db, encrypted_query



    def run_computation(self, encrypted_db, encrypted_query, vector_dim):
        """
        Computes cosine similarity (dot product for normalized vectors)
        and finds the entry with the highest score.
        """

        print("Step 3: Preparing for Dot Product computation (since vectors are normalized)...")
        sign_coeffs = get_sign_poly_coeffs() # Still needed for comparison
        print("\nStep 4: Computing encrypted dot products for each database entry...")
        similarities = []
        for i, enc_vec in enumerate(encrypted_db):
            # Compute Dot Product (A · B) - THIS IS THE COSINE SIMILARITY
            dot_product = self.cc.EvalMult(enc_vec, encrypted_query)
            self.cc.ModReduceInPlace(dot_product)
            dot_product_sum = self.cc.EvalSum(dot_product, vector_dim)
            similarities.append(dot_product_sum)
            print(f"   - Computed similarity for vector {i+1}/{len(encrypted_db)}")

        print(f"Computed {len(similarities)} similarity scores.")
        print("\nStep 5: Finding the encrypted maximum score...")
        scores = similarities
        while len(scores) > 1:
            next_round_scores = []

            for i in range(0, len(scores), 2):
                if i + 1 < len(scores):
                    max_val = _encrypted_compare(self.cc, scores[i], scores[i+1], sign_coeffs)
                    next_round_scores.append(max_val)
                else:
                    next_round_scores.append(scores[i])

            scores = next_round_scores

        C_max = scores[0]
        print("Encrypted maximum value computed.\n")
        return C_max

    def check_against_threshold(self, C_max, tau):
        print(f"Step 5: Performing threshold check against tau = {tau}...")
        C_diff = self.cc.EvalSub(C_max, tau)
        sign_coeffs = get_sign_poly_coeffs()
        C_sign = self.cc.EvalPoly(C_diff, sign_coeffs)
        print("Encrypted threshold check complete.\n")
        return C_sign

    def decrypt_result(self, ciphertext):
        """Performs a multiparty decryption of a ciphertext.
            This process involves three steps:

            1. The "lead" party (party 0) creates the first partial decryption.

            2. All other "main" parties create their own partial decryptions.

            3. The partial decryptions are fused to recover the final plaintext.
        """

        print("Step 6: Performing multiparty decryption...")
        partial_decryptions = []
        lead_secret_key = self.parties[0].kpShard.secretKey
        decryption_lead = self.cc.MultipartyDecryptLead([ciphertext], lead_secret_key)
        partial_decryptions.append(decryption_lead[0])
        print(" - Lead party (Party 0) provided partial decryption.")
        for i in range(1, len(self.parties)):
            main_secret_key = self.parties[i].kpShard.secretKey
            decryption_main = self.cc.MultipartyDecryptMain([ciphertext], main_secret_key)
            partial_decryptions.append(decryption_main[0])
            print(f"  - Main party (Party {i}) provided partial decryption.")
        # Fuse all partial decryptions to get the final plaintext result
        plaintext = self.cc.MultipartyDecryptFusion(partial_decryptions)
        print("Decryption done")
        # Set the length to control how many decrypted values are displayed,
        # then extract and return the numerical result.
        plaintext.SetLength(8) # Adjust as needed
        return plaintext.GetCKKSPackedValue()

def main():
    start_time = time.time()
    # Configuration
    NUM_VECTORS = 4
    DIMENSIONS = 512
    TAU = 0.85 # Threshold for unique/not-unique decision
    # Generate Plaintext Data 
    db, query = generate_data(NUM_VECTORS, DIMENSIONS)
    #  Initialization
    engine = PrivacyEngine(num_parties=3, ring_dim=1<<17, mult_depth=22, scaling_mod_size=58)
    encrypted_db, encrypted_query = engine.encrypt_data(db, query)
    C_max = engine.run_computation(encrypted_db, encrypted_query, DIMENSIONS)
    C_sign = engine.check_against_threshold(C_max, TAU)

    # Decryption
    decrypted_sign_result = engine.decrypt_result(C_sign)
    final_sign = decrypted_sign_result[0].real # The result is in the first slot
    print("Final Decision: ")

    if final_sign > 0:
        print(f"Decision: NOT-UNIQUE (Max similarity > {TAU})")
    else:
        print(f"Decision: UNIQUE (Max similarity <= {TAU})")

    # Accuracy CHeck
    print(" Accuracy CHeck")

    # Decrypt the actual max similarity score
    decrypted_max_result = engine.decrypt_result(C_max)
    he_max_similarity = decrypted_max_result[0].real

    # Compute the ground truth in plaintext
    plaintext_similarities = [np.dot(db_vec, query) for db_vec in db]
    plaintext_max_similarity = max(plaintext_similarities)

    # Calculate and report the error
    error = abs(he_max_similarity - plaintext_max_similarity)
    print(f"Plaintext Max Similarity: {plaintext_max_similarity:.8f}")
    print(f"Decrypted HE Max Similarity: {he_max_similarity:.8f}")
    print(f"Absolute Error: {error:.8f}")
    if error < 1e-4:
        print("Accuracy target met 10^-4.")
    else:
        print("Accuracy target not met.")

    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()
