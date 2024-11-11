MOD = 10**9 + 7

def countBalancedPermutations(num: str) -> int:
    from math import ceil, floor

    n = len(num)
    freq = [0] * 10
    total_sum = 0
    for ch in num:
        digit = int(ch)
        freq[digit] += 1
        total_sum += digit

    if total_sum % 2 != 0:
        return 0

    target_sum = total_sum // 2
    e = ceil(n / 2)
    o = n // 2

    # Precompute factorial and inverse factorial
    fact = [1] * (n + 1)
    for i in range(1, n + 1):
        fact[i] = fact[i - 1] * i % MOD

    inv_fact = [1] * (n + 1)
    inv_fact[n] = pow(fact[n], MOD - 2, MOD)
    for i in range(n - 1, -1, -1):
        inv_fact[i] = inv_fact[i + 1] * (i + 1) % MOD

    # Compute inv_fact_all = product of inv_fact[freq_i] for i in 0-9
    inv_fact_all = 1
    for f in freq:
        inv_fact_all = inv_fact_all * inv_fact[f] % MOD


    # Initialize DP
    dp = [ [0] * (target_sum + 1) for _ in range(e + 1) ]
    dp[0][0] = 1

    for digit in range(10):
        f = freq[digit]
        if f == 0:
            continue
        # Precompute combinations C(f, c) for c in 0 to f
        C = [0] * (f + 1)
        for c in range(0, f + 1):
            C[c] = fact[f] * inv_fact[c] % MOD
            C[c] = C[c] * inv_fact[f - c] % MOD
        # Update DP
        for k in range(e, -1, -1):
            for d in range(target_sum, -1, -1):
                if dp[k][d] == 0:
                    continue
                for c in range(1, min(f, e - k) + 1):
                    d_new = d + c * digit
                    if d_new > target_sum:
                        continue
                    dp[k + c][d_new] = (dp[k + c][d_new] + dp[k][d] * C[c]) % MOD

    X = dp[e][target_sum]

    result = fact[e] * fact[o] % MOD
    result = result * X % MOD
    result = result * inv_fact_all % MOD

    return result


# Example 1
print(countBalancedPermutations("123"))  # Output: 2

# Example 2
print(countBalancedPermutations("112"))  # Output: 1

# Example 3
print(countBalancedPermutations("12345"))  # Output: 0
