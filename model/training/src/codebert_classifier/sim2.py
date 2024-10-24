
print()
textTwo=input()
def jaccard_similarity(text1, text2):
    # Tokenize the texts
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    # Calculate Jaccard similarity
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    jaccard_sim = len(intersection) / len(union) if len(union) > 0 else 0

    # Convert to percentage (0 to 100)
    return jaccard_sim * 100


# Example texts
textOne=" ''' "+textOne+" ''' "
textTwo=" ''' "+textTwo+" ''' "
text1 =textOne
text2 =textTwo
similarity = jaccard_similarity(text1, text2)
print(f"Jaccard Similarity: {similarity:.2f}%")
# tweak gradient accumulation steps

# reformat code to pep8 standards