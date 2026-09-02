from typing import List
from collections import defaultdict


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        tokens = list(corpus)
        merges = []

        for _ in range(num_merges):
            # a. Count frequency of all adjacent token pairs
            pair_counts = defaultdict(int)
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_counts[pair] += 1

            if not pair_counts:
                break

            # b. Find the most frequent pair (break ties lexicographically)
            best_pair = min(
                pair_counts.keys(),
                key=lambda p: (-pair_counts[p], p)
            )

            # d. Record the merge as [token_a, token_b]
            merges.append([best_pair[0], best_pair[1]])

            # c. Merge all non-overlapping occurrences left to right
            merged_token = best_pair[0] + best_pair[1]
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (
                    i < len(tokens) - 1
                    and tokens[i] == best_pair[0]
                    and tokens[i + 1] == best_pair[1]
                ):
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens

        return merges